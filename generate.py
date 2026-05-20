"""
Generate text from a trained GPT checkpoint.

Usage:
    python generate.py --checkpoint checkpoints/best.pt --prompt "The universe began"
    python generate.py --checkpoint checkpoints/best.pt --prompt "In the city of" --temperature 1.0 --top_k 100 --max_tokens 300
"""

import argparse
import torch
from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a GPT checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--prompt",     type=str, default="Once upon a time",)
    p.add_argument("--max_tokens", type=int,   default=200)
    p.add_argument("--temperature",type=float, default=0.8,
                   help="Sampling temperature. Higher = more random (try 0.5–1.2)")
    p.add_argument("--top_k",      type=int,   default=50,
                   help="Keep top-k logits before sampling. 0 = disabled")
    p.add_argument("--num_samples",type=int,   default=1)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # ── Load checkpoint ──
    print(f"Loading {args.checkpoint}…")
    ckpt = torch.load(args.checkpoint, map_location=device)

    cfg = ckpt.get("config", GPTConfig())
    model = GPT(cfg)
    # Strip any torch.compile prefix
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ── Tokenizer ──
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except ImportError:
        raise ImportError("pip install transformers")

    # ── Generate ──
    ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, max_tokens={args.max_tokens}\n")
    print("─" * 60)

    for i in range(args.num_samples):
        out = model.generate(
            idx.clone(),
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        text = tokenizer.decode(out[0].tolist())
        print(f"[Sample {i+1}]\n{text}\n")
        print("─" * 60)


if __name__ == "__main__":
    main()
