"""Transformer asosidagi siyosat + WDL qiymat tarmog'i.
Arxitektura:
  - Har bir qora katak = token (32 token) + 1 CLS token (global holat)
  - Rotary Positional Embeddings (RoPE)
  - Multi-Head Attention, Pre-LayerNorm, Residual, GELU
  - Policy head: bilinear (from-vektor x to-vektor) -> 32x32 = 1024 logit
  - WDL head: 3 logit (Win / Draw / Loss) — durangga moyil shashka uchun
    skalyar qiymatdan ancha aniqroq kalibrlangan baho beradi.
ONNX export: kirish "x" [B, 194] float32,
             chiqish "policy_logits" [B, 1024], "wdl_logits" [B, 3].
Qiymat: v = P(win) - P(loss) (softmax(wdl) dan). """

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict, Tuple
from checkers_engine import ACTION_SIZE, INPUT_SIZE, NUM_SQUARES

class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 64, base: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim juft bo'lishi kerak"
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos_t", freqs.cos(), persistent=False)
        self.register_buffer("sin_t", freqs.sin(), persistent=False)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[-2]
        cos = self.cos_t[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_t[:T].unsqueeze(0).unsqueeze(0)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: RoPE) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rope = rope
        self.store_attn: bool = False
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.softmax(dim=-1)
        if self.store_attn:
            self.last_attn = att.detach()
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: RoPE) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
        nn.Linear(d_model, 4 * d_model),
        nn.GELU(),
        nn.Linear(4 * d_model, d_model),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ShashkaNet(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 6, n_heads: int = 8, policy_dim: int = 64) -> None:
        super().__init__()
        self.hparams: Dict[str, int] = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads, policy_dim=policy_dim)
        self.square_proj = nn.Linear(6, d_model)
        self.glob_proj = nn.Linear(2, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        rope = RoPE(d_model // n_heads, max_len=NUM_SQUARES + 2)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, rope) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.from_head = nn.Linear(d_model, policy_dim)
        self.to_head = nn.Linear(d_model, policy_dim)
        self.wdl_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        self.policy_scale = 1.0 / math.sqrt(policy_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        sq = x[:, :NUM_SQUARES * 6].reshape(B, NUM_SQUARES, 6)
        g = x[:, NUM_SQUARES * 6:]
        tok = self.square_proj(sq)
        cls = self.cls.expand(B, -1, -1) + self.glob_proj(g).unsqueeze(1)
        h = torch.cat([cls, tok], dim=1)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        sqh = h[:, 1:]
        fvec = self.from_head(sqh)
        tvec = self.to_head(sqh)
        logits = torch.einsum("bid,bjd->bij", fvec, tvec) * self.policy_scale
        return logits.reshape(B, ACTION_SIZE), self.wdl_head(h[:, 0])

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

def export_onnx(model: ShashkaNet, path: str) -> None:
    import glob
    import os
    import shutil
    import tempfile
    model = model.cpu().eval()
    dummy = torch.zeros(2, INPUT_SIZE, dtype=torch.float32)
    names = dict(input_names=["x"], output_names=["policy_logits", "wdl_logits"])
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmpdir = tempfile.mkdtemp(dir=directory)
    base = os.path.basename(path)
    tmp = os.path.join(tmpdir, base)
    try:
        try:
            batch = torch.export.Dim("batch", min=1, max=8192)
            torch.onnx.export(model, (dummy,),
            tmp, **names, dynamic_shapes={"x": {0: batch}}, opset_version=18, dynamo=True)
        except Exception:
            torch.onnx.export(model, (dummy,), tmp, **names,
            dynamic_axes={"x": {0: "batch"}, "policy_logits": {0: "batch"},
            "wdl_logits": {0: "batch"}}, opset_version=17, dynamo=False)
        for old in glob.glob(path + "*"):
            if os.path.isfile(old):
                os.remove(old)
        for produced in os.listdir(tmpdir):
            os.replace(os.path.join(tmpdir, produced), os.path.join(directory, produced))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)