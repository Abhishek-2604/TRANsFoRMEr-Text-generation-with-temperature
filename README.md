# GPT Language Model — From Scratch in PyTorch

A clean, well-commented implementation of a GPT-style autoregressive transformer
for language modeling, trained on WikiText-103.

---

## Architecture

```
Input tokens
    │
    ▼
Token Embeddings (vocab_size → n_embd)
    +
Position Embeddings (context_length → n_embd)
    │
    ▼
Dropout
    │
    ▼  ┌─────────────────────────────────────────────┐
    │  │  Transformer Block  × n_layer               │
    │  │                                             │
    │  │  x = x + CausalSelfAttention(LayerNorm(x)) │
    │  │  x = x + MLP(LayerNorm(x))                  │
    │  └─────────────────────────────────────────────┘
    │
    ▼
Final LayerNorm
    │
    ▼
LM Head (n_embd → vocab_size)  [weights tied to token embeddings]
    │
    ▼
Cross-Entropy Loss (training) / Softmax Sampling (inference)
```

### Default config (~22M parameters)
| Param            | Value |
|------------------|-------|
| `n_layer`        | 6     |
| `n_head`         | 6     |
| `n_embd`         | 384   |
| `context_length` | 256   |
| `dropout`        | 0.1   |
| `vocab_size`     | 50257 |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
python train.py
```

The first run downloads and tokenizes WikiText-103 into `data/` (one-time, ~500MB).
Checkpoints are saved to `checkpoints/`.

### 3. Generate text
```bash
python generate.py --checkpoint checkpoints/best.pt --prompt "The history of science"
```

---

## Training options

```
--n_layer        Number of transformer blocks         (default: 6)
--n_head         Number of attention heads            (default: 6)
--n_embd         Embedding dimension                  (default: 384)
--context_length Sequence length                      (default: 256)
--dropout        Dropout rate                         (default: 0.1)
--batch_size     Batch size                           (default: 32)
--max_iters      Total training steps                 (default: 10000)
--lr             Peak learning rate                   (default: 3e-4)
--warmup_iters   LR warmup steps                      (default: 500)
--weight_decay   AdamW weight decay                   (default: 0.1)
--grad_clip      Gradient clipping                    (default: 1.0)
--eval_interval  Steps between evaluations            (default: 500)
--out_dir        Checkpoint directory                 (default: checkpoints)
--resume         Resume from checkpoint               (default: None)
```

Example — faster, smaller run:
```bash
python train.py --n_layer 4 --n_embd 256 --n_head 4 --max_iters 5000 --batch_size 64
```

---

## Generation options

```
--checkpoint   Path to .pt checkpoint          (required)
--prompt       Conditioning text               (default: "Once upon a time")
--max_tokens   Tokens to generate              (default: 200)
--temperature  Sampling temperature 0.5–1.2    (default: 0.8)
--top_k        Top-k filtering                 (default: 50)
--num_samples  Number of independent samples   (default: 1)
```

---

## File layout

```
transformer_lm/
├── model.py          # GPT architecture (attention, MLP, blocks)
├── data.py           # Data loading & tokenization
├── train.py          # Training loop, LR schedule, checkpointing
├── generate.py       # Text generation from checkpoint
├── requirements.txt
└── README.md
```

---

## Key design choices

- **Pre-norm**: LayerNorm applied *before* attention/MLP (more stable than post-norm)
- **Weight tying**: Token embedding and output projection share weights (~saves 20M params)
- **AdamW decoupled decay**: Weight decay applied only to weight matrices, not biases or norms
- **Cosine LR schedule**: Linear warmup + cosine annealing to `min_lr`
- **Mixed precision**: `torch.autocast` on CUDA for ~2× speed
- **Causal mask**: Lower-triangular mask prevents tokens attending to future positions
