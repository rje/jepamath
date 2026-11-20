"""Surprise from a frozen V-JEPA 2 (encoder + pretraining predictor), following Garrido et al. 2025.

For each 16-frame window, the context encoder sees only the first `ctx` frames; the predictor predicts
the remaining tokens; the target is the same encoder on the full window, layer-normed over features.
Surprise of a window = mean L1 between prediction and target over predicted tokens and features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .guard import ComputeGuard

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TUBELET = 2


class VJepaSurprise:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", dtype=torch.bfloat16,
                 guard: ComputeGuard | None = None, window_batch: int = 16):
        from transformers import VJEPA2Model
        self.model = VJEPA2Model.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        cfg = self.model.config
        self.res = cfg.crop_size
        self.grid = cfg.crop_size // cfg.patch_size
        self.device, self.dtype = device, dtype
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
        T = video.shape[1]
        if T < window:
            raise ValueError(f"video has {T} frames < window {window}")
        pieces = video.unfold(1, window, stride).permute(1, 4, 0, 2, 3)   # [nW, window, 3, H, W] (HF layout)
        tpf = self.grid * self.grid
        N = (window // TUBELET) * tpf
        out = []
        for c in context_lengths:
            n_ctx = (c // TUBELET) * tpf
            losses = []
            for k in range(0, pieces.shape[0], self.window_batch):
                x = pieces[k:k + self.window_batch].to(self.dtype)
                B = x.shape[0]
                with self.guard:
                    h = self.model.encoder(x).last_hidden_state.float()
                    h = F.layer_norm(h, (h.size(-1),))
                    tgt = h[:, n_ctx:]
                    z = self.model.encoder(x[:, :c]).last_hidden_state
                    if z.shape[1] != n_ctx:
                        raise RuntimeError(f"context encoding has {z.shape[1]} tokens, expected {n_ctx}")
                    cm = [torch.arange(n_ctx, device=self.device).unsqueeze(0).repeat(B, 1)]
                    tm = [torch.arange(n_ctx, N, device=self.device).unsqueeze(0).repeat(B, 1)]
                    pred = self.model.predictor(encoder_hidden_states=z, context_mask=cm, target_mask=tm).last_hidden_state
                    losses.append((pred.float() - tgt).abs().mean((1, 2)).cpu())
            out.append(torch.cat(losses))
        return torch.stack(out)
