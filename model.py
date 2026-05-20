"""
GPT-style Transformer for Language Modeling
Built from scratch with PyTorch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    vocab_size: int = 50257       # GPT-2 BPE vocab
    context_length: int = 256     # max sequence length
    n_layer: int = 6              # number of transformer blocks
    n_head: int = 6               # attention heads
    n_embd: int = 384             # embedding dimension
    dropout: float = 0.1
    bias: bool = True             # bias in linear layers & layernorm

    @property
    def head_dim(self):
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (nn.LayerNorm always uses bias)."""
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # Q, K, V projections packed into one matrix for efficiency
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Causal mask: lower-triangular so token i only attends to tokens <= i
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_length, cfg.context_length))
            .view(1, 1, cfg.context_length, cfg.context_length),
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim
        H = self.cfg.n_head
        D = self.cfg.head_dim

        # Compute Q, K, V and split into heads
        q, k, v = self.c_attn(x).split(self.cfg.n_embd, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(D)
        attn = (q @ k.transpose(-2, -1)) * scale          # (B, H, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate values and project
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(out))


class MLP(nn.Module):
    """Position-wise feed-forward network (2-layer MLP with GELU)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """Transformer block: LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # pre-norm residual
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-style autoregressive language model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),        # token embeddings
            wpe  = nn.Embedding(cfg.context_length, cfg.n_embd),    # position embeddings
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding & output projection weights
        self.transformer.wte.weight = self.lm_head.weight

        self._init_weights()
        print(f"GPT model — {self.num_params():,} parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue
            nn.init.normal_(p, mean=0.0, std=0.02)
            # Scale residual projections by 1/sqrt(2 * n_layer) per GPT-2 paper
            if name.endswith(("c_proj.weight", "proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layer))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.cfg.context_length, f"Sequence length {T} > context_length {self.cfg.context_length}"

        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss

        # Inference: only compute logits for the last token (efficiency)
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with temperature + top-k sampling."""
        for _ in range(max_new_tokens):
            # Crop context to max length
            idx_cond = idx if idx.size(1) <= self.cfg.context_length else idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    def configure_optimizers(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        """AdamW with weight decay only on 2D parameters (not biases/norms)."""
        decay_params = [p for n, p in self.named_parameters() if p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.dim() < 2]
        groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        print(f"Optimizer — decay: {sum(p.numel() for p in decay_params):,} params | "
              f"no-decay: {sum(p.numel() for p in nodecay_params):,} params")
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=False)
