"""
Training loop for the GPT language model.

Features:
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Periodic evaluation + perplexity logging
  - Checkpoint save/resume
  - Text generation samples during training
"""

import os
import time
import math
import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig
from data import make_loaders


# ─── Hyperparameters ─────────────────────────────────────────────────────────

def get_config():
    p = argparse.ArgumentParser(description="Train a GPT language model")

    # Model
    p.add_argument("--n_layer",        type=int,   default=6)
    p.add_argument("--n_head",         type=int,   default=6)
    p.add_argument("--n_embd",         type=int,   default=384)
    p.add_argument("--context_length", type=int,   default=256)
    p.add_argument("--dropout",        type=float, default=0.1)

    # Training
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--max_iters",     type=int,   default=10_000)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--min_lr",        type=float, default=3e-5)
    p.add_argument("--warmup_iters",  type=int,   default=500)
    p.add_argument("--weight_decay",  type=float, default=0.1)
    p.add_argument("--grad_clip",     type=float, default=1.0)

    # Eval / logging
    p.add_argument("--eval_interval",    type=int, default=500)
    p.add_argument("--eval_iters",       type=int, default=50)
    p.add_argument("--log_interval",     type=int, default=100)
    p.add_argument("--generate_interval",type=int, default=1000)
    p.add_argument("--sample_prompt",    type=str, default="Once upon a time")

    # I/O
    p.add_argument("--data_dir",   type=str, default="data")
    p.add_argument("--out_dir",    type=str, default="checkpoints")
    p.add_argument("--resume",     type=str, default=None, help="path to checkpoint to resume")
    p.add_argument("--num_workers",type=int, default=0)

    return p.parse_args()


# ─── LR Schedule ─────────────────────────────────────────────────────────────

def get_lr(step: int, cfg) -> float:
    """Linear warmup + cosine decay."""
    if step < cfg.warmup_iters:
        return cfg.lr * step / cfg.warmup_iters
    if step > cfg.max_iters:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ─── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    model.eval()
    out = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        it = iter(loader)
        for _ in range(eval_iters):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        avg = sum(losses) / len(losses)
        out[split] = avg
    model.train()
    return out


# ─── Generation sample ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str, device, max_new_tokens=100, temperature=0.8, top_k=40):
    model.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = tokenizer.decode(out[0].tolist())
    model.train()
    return text


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train():
    args = get_config()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Compile model for speed on PyTorch 2.x
    use_compile = hasattr(torch, "compile") and device == "cuda"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Data ──
    train_loader, val_loader = make_loaders(
        args.data_dir, args.context_length, args.batch_size, args.num_workers
    )
    train_iter = iter(train_loader)

    # ── Tokenizer (for generation samples) ──
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception:
        tokenizer = None

    # ── Model ──
    cfg = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        context_length=args.context_length,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)

    if use_compile:
        print("Compiling model with torch.compile…")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(args.lr, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_iter = 0
    best_val_loss = float("inf")

    # ── Resume ──
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # Handle compiled model state dict
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"]
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"  Resumed at iter {start_iter}, best val loss {best_val_loss:.4f}")

    # ── Training ──
    model.train()
    t0 = time.time()

    for step in range(start_iter, args.max_iters):

        # LR update
        lr = get_lr(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Fetch batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # Forward + backward
        with torch.autocast(device_type=device if device != "mps" else "cpu",
                            dtype=torch.float16 if device == "cuda" else torch.float32,
                            enabled=(device == "cuda")):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ── Logging ──
        if step % args.log_interval == 0:
            dt = time.time() - t0
            tokens_per_sec = (args.log_interval * args.batch_size * args.context_length) / dt
            print(f"iter {step:6d} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {tokens_per_sec:,.0f} tok/s")
            t0 = time.time()

        # ── Evaluation ──
        if step % args.eval_interval == 0 and step > 0:
            losses = estimate_loss(model, train_loader, val_loader, args.eval_iters, device)
            val_ppl = math.exp(min(losses["val"], 20))  # cap to avoid overflow
            print(f"\n{'─'*60}")
            print(f"EVAL iter {step} | train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | val ppl {val_ppl:.2f}")
            print(f"{'─'*60}\n")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt = {
                    "iter": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                }
                ckpt_path = out_dir / "best.pt"
                torch.save(ckpt, ckpt_path)
                print(f"  ✓ New best ({best_val_loss:.4f}) — saved to {ckpt_path}")

        # ── Generation sample ──
        if step % args.generate_interval == 0 and step > 0 and tokenizer:
            print("\n── Sample generation ──")
            text = generate_sample(model, tokenizer, args.sample_prompt, device)
            print(text)
            print("────────────────────\n")

    # Final checkpoint
    ckpt_path = out_dir / "final.pt"
    torch.save({"iter": args.max_iters, "model": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"\nTraining complete. Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train()
