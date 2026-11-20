"""Surprise from a frozen V-JEPA 2 (encoder + pretraining predictor), following Garrido et al. 2025.

For each 16-frame window, the context encoder sees only the first `ctx` frames; the predictor predicts
the remaining tokens; the target is the EMA target encoder on the full window, layer-normed over features.
The HF checkpoint ships only the context encoder, so the target encoder is loaded from Meta's original
checkpoint (`official_ckpt`) through a key mapping that is verified against the shipped context encoder.
Surprise of a window = mean L1 between prediction and target over predicted tokens and features.
"""

from __future__ import annotations

import copy
import functools
import re

import torch
import torch.nn.functional as F

from .guard import ComputeGuard

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TUBELET = 2
MASK_INDEX = 0
OFFICIAL_CKPT = "/mnt/data/ext/ckpt/vitl.pt"   # https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
_QKV = re.compile(r"layer\.(\d+)\.attention\.(query|key|value)\.(weight|bias)")


def _official_key_map(hf_sd: dict, off_ctx: dict) -> dict:
    """HF encoder key -> (official key, qkv chunk or None), found by exact tensor equality with the
    official context encoder (which HF ships unchanged). Raises unless every key maps uniquely."""
    by_shape: dict = {}
    for k, v in off_ctx.items():
        by_shape.setdefault(tuple(v.shape), []).append(k)
    out = {}
    for k, v in hf_sd.items():
        m = _QKV.fullmatch(k)
        if m:
            i, part, kind = m.groups()
            ok = f"module.backbone.blocks.{i}.attn.qkv.{kind}"
            chunk = ("query", "key", "value").index(part)
            if not torch.equal(off_ctx[ok].to(v.dtype).chunk(3, 0)[chunk], v):
                raise RuntimeError(f"qkv split mismatch for {k}")
            out[k] = (ok, chunk)
            continue
        # compare in the HF model's dtype: an fp32 original casts to exactly the shipped bf16 tensor
        cands = [ok for ok in by_shape.get(tuple(v.shape), []) if torch.equal(off_ctx[ok].to(v.dtype), v)]
        if len(cands) != 1:
            raise RuntimeError(f"{k}: {len(cands)} exact matches in the official encoder")
        out[k] = (cands[0], None)
    return out


def load_target_encoder(hf_encoder: torch.nn.Module, ckpt_path: str = OFFICIAL_CKPT) -> torch.nn.Module:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hf_sd = {k: v.detach().cpu() for k, v in hf_encoder.state_dict().items()}
    kmap = _official_key_map(hf_sd, ck["encoder"])
    tgt = ck["target_encoder"]
    new_sd = {}
    for k, (ok, chunk) in kmap.items():
        w = tgt[ok].float()
        new_sd[k] = (w if chunk is None else w.chunk(3, 0)[chunk]).to(hf_sd[k].dtype)
    te = copy.deepcopy(hf_encoder)
    te.load_state_dict(new_sd, strict=True)
    diff = max(float((new_sd[k].float() - hf_sd[k].float()).abs().max()) for k in new_sd)
    if diff == 0:
        raise RuntimeError("target encoder is identical to the context encoder; wrong checkpoint?")
    return te.eval()


def _check_precision(precision: str):
    # weights stay fp32; "fp16"/"bf16" run under autocast (Garrido used fp16 autocast), "fp32" runs plain
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision must be fp32, fp16 or bf16, got {precision!r}")
    return {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


class _SurpriseBase:
    """Shared windowing/preprocessing. Subclasses set res, grid, max_window and implement _batch(x, c),
    where x is [B, 3, window, res, res] and the return is [B, n_target_tokens] per-token surprise (L1 averaged
    over features) for context length c frames; target tokens are ordered time slot major."""
    res: int
    grid: int
    max_window: int | None = None

    def _setup(self, device, precision, guard, window_batch):
        self.autocast = _check_precision(precision)
        self.device = device
        self.guard = guard or ComputeGuard()
        self.window_batch = window_batch
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1, 1)

    def preprocess(self, video) -> torch.Tensor:
        """uint8 [T, H, W, 3] (numpy or tensor) -> float [3, T, res, res], ImageNet-normalised."""
        v = torch.as_tensor(video, device=self.device)
        if v.dtype != torch.uint8 or v.ndim != 4 or v.shape[-1] != 3:
            raise ValueError(f"expected uint8 [T,H,W,3], got {v.dtype} {tuple(v.shape)}")
        v = v.permute(0, 3, 1, 2).float() / 255.0                     # [T,3,H,W]
        if v.shape[-2:] != (self.res, self.res):
            if v.shape[-1] != v.shape[-2]:
                raise ValueError(f"non-square frames {tuple(v.shape[-2:])}; crop before calling")
            v = F.interpolate(v, size=(self.res, self.res), mode="bilinear", antialias=True, align_corners=False)
        v = v.permute(1, 0, 2, 3).unsqueeze(0)                         # [1,3,T,H,W]
        return ((v - self.mean) / self.std)[0]

    @torch.no_grad()
    def window_losses(self, video: torch.Tensor, context_lengths=(2, 4, 6, 8, 10), window: int = 16,
                      stride: int = 2) -> torch.Tensor:
        """video: preprocessed [3, T, res, res]. Returns [len(context_lengths), n_windows] surprise.
        Window w covers frames [w*stride, w*stride + window)."""
        if window % TUBELET or any(c % TUBELET or not 0 < c < window for c in context_lengths):
            raise ValueError("window and context lengths must be multiples of the tubelet size, 0 < ctx < window")
        if self.max_window is not None and window != self.max_window:
            raise ValueError(f"this model takes exactly {self.max_window}-frame windows")
        T = video.shape[1]
        if T < window:
            raise ValueError(f"video has {T} frames < window {window}")
        pieces = video.unfold(1, window, stride).permute(1, 0, 4, 2, 3)   # [nW, 3, window, H, W]
        out = []
        for c in context_lengths:
            losses = []
            for k in range(0, pieces.shape[0], self.window_batch):
                x = pieces[k:k + self.window_batch]
                with self.guard, torch.autocast("cuda", dtype=self.autocast or torch.float16,
                                                enabled=self.autocast is not None):
                    losses.append(self._batch(x, c).float().mean(1).cpu())
            out.append(torch.cat(losses))
        return torch.stack(out)

    @torch.no_grad()
    def slot_losses(self, video: torch.Tensor, ctx: int) -> torch.Tensor:
        """One window = the whole preprocessed clip [3, T, res, res]; the context is its first `ctx` frames.
        Returns [n_target_slots] surprise per predicted time slot (tubelet), slot j covering frames
        [ctx + 2j, ctx + 2j + 2)."""
        T = video.shape[1]
        if self.max_window is not None and T != self.max_window:
            raise ValueError(f"this model takes exactly {self.max_window}-frame clips, got {T}")
        if T % TUBELET or ctx % TUBELET or not 0 < ctx < T:
            raise ValueError(f"bad clip/context: T={T} ctx={ctx}")
        with self.guard, torch.autocast("cuda", dtype=self.autocast or torch.float16,
                                        enabled=self.autocast is not None):
            tok = self._batch(video.unsqueeze(0), ctx).float()[0]
        return tok.view((T - ctx) // TUBELET, self.grid * self.grid).mean(1).cpu()


class VJepaSurprise(_SurpriseBase):
    """V-JEPA 2 via Hugging Face, with the official target encoder."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", precision: str = "fp16",
                 guard: ComputeGuard | None = None, window_batch: int = 16, official_ckpt: str | None = OFFICIAL_CKPT):
        from transformers import VJEPA2Model
        self._setup(device, precision, guard, window_batch)
        self.model = VJEPA2Model.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
        cfg = self.model.config
        # The HF predictor always uses mask token 1, but the released V-JEPA 2 checkpoints trained only
        # token 0 (tokens 1..9 are zeros). Pin the index and refuse checkpoints where it is untrained.
        emb = self.model.predictor.embeddings
        tokens = emb.mask_tokens.detach()
        if not tokens[MASK_INDEX].abs().sum() > 0:
            raise RuntimeError(f"{model_id}: predictor mask token {MASK_INDEX} is all zeros (untrained)")
        emb.forward = functools.partial(emb.forward, mask_index=MASK_INDEX)
        # official_ckpt=None uses the context encoder for targets too (ablation only, not the protocol)
        self.target_encoder = (load_target_encoder(self.model.encoder, official_ckpt).to(device)
                               if official_ckpt else self.model.encoder)
        self.res = cfg.crop_size
        self.grid = cfg.crop_size // cfg.patch_size

    def _batch(self, x, c):
        x = x.permute(0, 2, 1, 3, 4)                                   # HF layout [B, T, 3, H, W]
        B, window = x.shape[0], x.shape[1]
        tpf = self.grid * self.grid
        n_ctx, N = (c // TUBELET) * tpf, (window // TUBELET) * tpf
        h = self.target_encoder(x).last_hidden_state.float()
        tgt = F.layer_norm(h, (h.size(-1),))[:, n_ctx:]
        z = self.model.encoder(x[:, :c]).last_hidden_state
        if z.shape[1] != n_ctx:
            raise RuntimeError(f"context encoding has {z.shape[1]} tokens, expected {n_ctx}")
        cm = [torch.arange(n_ctx, device=self.device).unsqueeze(0).repeat(B, 1)]
        tm = [torch.arange(n_ctx, N, device=self.device).unsqueeze(0).repeat(B, 1)]
        pred = self.model.predictor(encoder_hidden_states=z, context_mask=cm, target_mask=tm).last_hidden_state
        return (pred.float() - tgt).abs().mean(2)


VJEPA1_REPO = "/mnt/data/ext/jepa1"                       # github.com/facebookresearch/jepa
VJEPA1_CKPT = "/mnt/data/ext/ckpt/vjepa1_vitl16.pth.tar"  # dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar


class VJepa1Surprise(_SurpriseBase):
    """Official V-JEPA 1 ViT-L/16 (224 px, 16 frames, absolute pos-embed) through Meta's code; this is the
    model family Garrido et al. report, so it validates the pipeline. Mask token: predictor default (1),
    as in Garrido's eval. Never import alongside the V-JEPA 2 repo (both use a top-level `src`)."""
    max_window = 16

    def __init__(self, device: str = "cuda", precision: str = "fp16", guard: ComputeGuard | None = None,
                 window_batch: int = 16, repo: str = VJEPA1_REPO, ckpt: str = VJEPA1_CKPT):
        import sys
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.models import predictor as vp, vision_transformer as ve
        self._setup(device, precision, guard, window_batch)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        strip = lambda sd: {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        mk = lambda: ve.vit_large(img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
                                  uniform_power=True, use_sdpa=True)
        self.ctx_enc, self.tgt_enc = mk(), mk()
        self.ctx_enc.load_state_dict(strip(ck["encoder"]), strict=True)
        self.tgt_enc.load_state_dict(strip(ck["target_encoder"]), strict=True)
        self.pred = vp.vit_predictor(img_size=224, use_mask_tokens=True, patch_size=16, num_frames=16,
                                     tubelet_size=2, embed_dim=1024, predictor_embed_dim=384, depth=12,
                                     num_heads=16, uniform_power=True, num_mask_tokens=2,
                                     zero_init_mask_tokens=True, use_sdpa=True)
        self.pred.load_state_dict(strip(ck["predictor"]), strict=True)
        for m in (self.ctx_enc, self.tgt_enc, self.pred):
            m.to(device).eval()
        self.res, self.grid = 224, 14

    def _batch(self, x, c):
        B, window = x.shape[0], x.shape[2]
        tpf = self.grid * self.grid
        n_ctx, N = (c // TUBELET) * tpf, (window // TUBELET) * tpf
        h = self.tgt_enc(x).float()
        tgt = F.layer_norm(h, (h.size(-1),))[:, n_ctx:]
        cm = torch.arange(n_ctx, device=self.device).unsqueeze(0).repeat(B, 1)
        tm = torch.arange(n_ctx, N, device=self.device).unsqueeze(0).repeat(B, 1)
        z = self.ctx_enc(x, masks=[cm])
        pred = self.pred(z, None, [cm], [tm])
        return (pred.float() - tgt).abs().mean(2)
