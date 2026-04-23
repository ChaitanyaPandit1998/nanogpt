"""
DataLoaderLite: shard-based data loading from pre-tokenized .npy files.

Loads pre-tokenized shards from the edu_fineweb10B directory and distributes
them across DDP ranks. Each shard is a flat int32 numpy array; batches are
consecutive slices of length B*T+1 (inputs + one-token lookahead for targets).
"""

import os
import numpy as np
import torch


def load_tokens(filename: str) -> torch.Tensor:
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    """
    Minimal distributed data loader for pre-tokenized .npy shard files.

    Shards must live in edu_fineweb10B/ and contain 'train' or 'val' in their
    filename. Each process reads a different slice of the data (DDP-aware).
    Automatically advances to the next shard when the current one is exhausted.
    """

    def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: str):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        data_root = "edu_fineweb10B"
        shards = sorted(s for s in os.listdir(data_root) if split in s)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no shards found for split '{split}' in {data_root}/"
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets (shifted by one)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
