"""
Data pipeline for language modeling.
Uses WikiText-103 via HuggingFace datasets + GPT-2 BPE tokenizer.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def prepare_data(data_dir: str = "data", context_length: int = 256) -> tuple:
    """
    Download and tokenize WikiText-103. Saves binary files for fast reloading.
    Returns (train_dataset, val_dataset).
    """
    try:
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
    except ImportError:
        raise ImportError("pip install datasets transformers tiktoken")

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    train_bin = data_path / "train.bin"
    val_bin   = data_path / "val.bin"

    if not train_bin.exists():
        print("Downloading & tokenizing WikiText-103 (one-time setup)...")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        dataset = load_dataset("wikitext", "wikitext-103-v1")

        def tokenize(batch):
            ids = tokenizer(batch["text"])["input_ids"]
            # Flatten all ids from this batch into a single list
            flat = []
            for seq in ids:
                flat.extend(seq)
            return {"ids": flat}

        for split, fname in [("train", train_bin), ("validation", val_bin)]:
            print(f"  Tokenizing {split}...")
            tokens = []
            for example in dataset[split]:
                text = example["text"].strip()
                if text:
                    enc = tokenizer.encode(text)
                    tokens.extend(enc)
            arr = np.array(tokens, dtype=np.uint16)
            arr.tofile(fname)
            print(f"  {split}: {len(arr):,} tokens → {fname}")

    train_ds = TokenDataset(train_bin, context_length)
    val_ds   = TokenDataset(val_bin,   context_length)
    print(f"Train: {len(train_ds):,} chunks | Val: {len(val_ds):,} chunks")
    return train_ds, val_ds


class TokenDataset(Dataset):
    """
    Memory-mapped dataset of fixed-length token windows.
    Input: tokens[0..T-1], Target: tokens[1..T] (next-token prediction).
    """

    def __init__(self, bin_file: Path, context_length: int):
        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.context_length = context_length
        # Number of complete windows we can extract
        self.n = max(0, len(self.data) - context_length)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.context_length + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def make_loaders(
    data_dir: str = "data",
    context_length: int = 256,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple:
    train_ds, val_ds = prepare_data(data_dir, context_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader
