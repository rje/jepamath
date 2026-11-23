"""Small video JEPA (V-JEPA style): ViT encoder over tubelet tokens, EMA target encoder, narrow predictor.

Tokens are ordered time-slot major: index = slot * grid**2 + row * grid + col, so "the first c slots" is a
prefix of the token sequence. Positions are fixed 3D sin-cos, so an encoder fed a subset of tokens (by index)
sees their true positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

FAST_SDPA = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


def sincos_3d(t: int, h: int, w: int, dim: int) -> torch.Tensor:
    """[t*h*w, dim] fixed embedding; dim split 1/4 time, 3/8 rows, 3/8 cols (each an even count)."""
    dt = dim // 4 // 2 * 2
    dh = (dim - dt) // 2 // 2 * 2
    dw = dim - dt - dh
    if dw % 2:
        raise ValueError(f"dim {dim} does not split into even sin-cos parts")

    def one(n, d):
        pos = torch.arange(n, dtype=torch.float64)
        om = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2))
        a = pos[:, None] * om[None]
        return torch.cat([a.sin(), a.cos()], 1)

    et, eh, ew = one(t, dt), one(h, dh), one(w, dw)
    e = torch.cat([et[:, None, None].expand(t, h, w, dt), eh[None, :, None].expand(t, h, w, dh),
                   ew[None, None, :].expand(t, h, w, dw)], -1)
    return e.reshape(t * h * w, dim).float()


class Block(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if q.is_cuda:
            # fast kernels only: the math fallback is ~27x slower at 8k tokens (measured on the RTX 6000 Ada), and
            # silently landing on it would waste a run; with neither fast kernel available this raises instead
            with sdpa_kernel(FAST_SDPA):
                a = F.scaled_dot_product_attention(q, k, v)
        else:
            a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(a)
        return x + self.mlp(self.n2(x))


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    def __init__(self, img=128, patch=8, frames=32, tubelet=4, dim=384, depth=8, heads=6):
        super().__init__()
        if img % patch or frames % tubelet:
            raise ValueError("img/patch and frames/tubelet must divide")
        self.img, self.patch, self.frames, self.tubelet, self.dim = img, patch, frames, tubelet, dim
        self.grid, self.slots = img // patch, frames // tubelet
        self.embed = nn.Conv3d(3, dim, (tubelet, patch, patch), (tubelet, patch, patch))
        self.register_buffer("pos", sincos_3d(self.slots, self.grid, self.grid, dim), persistent=False)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.apply(_init)
        for i, b in enumerate(self.blocks):   # depth-scaled residual init
            b.proj.weight.data.div_(math.sqrt(2 * (i + 1)))
            b.mlp[2].weight.data.div_(math.sqrt(2 * (i + 1)))

    @property
    def tokens_per_slot(self):
        return self.grid * self.grid

    def forward(self, x, keep: torch.Tensor | None = None):
        """x [B, 3, T, H, W] with T <= frames (a prefix of the clip is fine); keep [B, K] token indices."""
        t = self.embed(x).flatten(2).transpose(1, 2)
        t = t + self.pos[: t.shape[1]]
        if keep is not None:
            t = torch.gather(t, 1, keep[..., None].expand(-1, -1, t.shape[-1]))
        for b in self.blocks:
            t = b(t)
        return self.norm(t)


class Predictor(nn.Module):
    def __init__(self, enc: Encoder, dim=256, depth=4, heads=8):
        super().__init__()
        self.inp = nn.Linear(enc.dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos", sincos_3d(enc.slots, enc.grid, enc.grid, dim), persistent=False)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, enc.dim)
        self.apply(_init)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, z, ctx_idx, tgt_idx):
        """z [B, Nc, enc_dim] encodings of tokens ctx_idx [B, Nc]; returns predictions [B, Nt, enc_dim] for tgt_idx."""
        x = self.inp(z) + self.pos[ctx_idx]
        m = self.mask_token + self.pos[tgt_idx]
        h = torch.cat([x, m.expand(z.shape[0], -1, -1)], 1)
        for b in self.blocks:
            h = b(h)
        return self.out(self.norm(h[:, x.shape[1]:]))
