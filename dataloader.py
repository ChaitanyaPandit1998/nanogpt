"""
DataLoaderLite: shard-based data loading from pre-tokenized .npy files.

Supports two modes:

Single-source (original, backward-compatible):
    DataLoaderLite(B, T, rank, world_size, split, data_root="edu_fineweb10B")

Multi-source weighted (new):
    DataLoaderLite(B, T, rank, world_size, split,
                   sources=[
                       ("/workspace/pretrain_data/fineweb/", 0.676),
                       ("/workspace/pretrain_data/sec/",     0.243),
                       ("/workspace/pretrain_data/code/",    0.081),
                   ])

In multi-source mode, each call to next_batch() picks a source proportionally
to its weight using a seeded RNG — deterministic and consistent across DDP ranks
(all ranks construct the loader identically and call next_batch() in lockstep).

For val split in multi-source mode, only the first source is used (fineweb val
shards). This keeps validation cost predictable.
"""

import os
import random
import numpy as np
import torch


def load_tokens(filename: str) -> torch.Tensor:
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


# ---------------------------------------------------------------------------
# Internal single-source loader — used by DataLoaderLite for each source


class _SourceLoader:
    """Loads batches from one directory of .npy shard files."""

    def __init__(self, B: int, T: int, process_rank: int, num_processes: int,
                 split: str, data_root: str):
        self.B = B
        self.T = T
        self.process_rank  = process_rank
        self.num_processes = num_processes
        self.split         = split
        self.data_root     = data_root

        shards = sorted(s for s in os.listdir(data_root) if split in s)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, \
            f"no shards found for split '{split}' in {data_root}/"
        self.reset()

    def reset(self):
        self.current_shard    = 0
        self.tokens           = load_tokens(self.shards[0])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens        = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    def state_dict(self) -> dict:
        return {
            "shard": self.current_shard,
            "pos":   self.current_position,
        }

    def load_state_dict(self, state: dict):
        self.current_shard    = state["shard"]
        self.tokens           = load_tokens(self.shards[state["shard"]])
        self.current_position = state["pos"]


# ---------------------------------------------------------------------------
# Public DataLoaderLite


class DataLoaderLite:
    """
    Shard-based data loader, single- or multi-source.

    Single-source (backward compatible):
        loader = DataLoaderLite(B, T, rank, world_size, split,
                                data_root="edu_fineweb10B")

    Multi-source weighted:
        loader = DataLoaderLite(B, T, rank, world_size, split,
                                sources=[
                                    ("/workspace/pretrain_data/fineweb/", 0.676),
                                    ("/workspace/pretrain_data/sec/",     0.243),
                                    ("/workspace/pretrain_data/code/",    0.081),
                                ])

    Attributes kept for backward compatibility:
        .shards             list of shard paths (primary source)
        .current_shard      current shard index (primary source)
        .current_position   current position (primary source)
        .tokens             current token tensor (primary source)
    """

    def __init__(self,
                 B: int, T: int,
                 process_rank: int, num_processes: int,
                 split: str,
                 data_root: str = None,
                 sources: list = None):
        """
        Args:
            B, T:            micro-batch size and sequence length
            process_rank:    DDP rank of this process
            num_processes:   total DDP world size
            split:           'train' or 'val'
            data_root:       single-source directory (backward compat)
            sources:         list of (directory, weight) tuples for multi-source
        """
        assert split in {"train", "val"}
        assert (data_root is None) != (sources is None), \
            "Provide exactly one of data_root (single-source) or sources (multi-source)"

        self.B             = B
        self.T             = T
        self.process_rank  = process_rank
        self.num_processes = num_processes
        self.split         = split

        if data_root is not None:
            # Single-source — original behaviour
            self._multi = False
            self._loaders = [_SourceLoader(B, T, process_rank, num_processes,
                                           split, data_root)]
            self._weights = [1.0]
        else:
            # Multi-source weighted
            self._multi = True
            dirs, weights = zip(*sources)
            total = sum(weights)
            self._weights = [w / total for w in weights]

            if split == "val":
                # Use only the first source for validation (predictable cost)
                self._loaders = [_SourceLoader(B, T, process_rank, num_processes,
                                               split, dirs[0])]
                self._weights = [1.0]
                self._multi   = False  # val acts as single-source
            else:
                self._loaders = [
                    _SourceLoader(B, T, process_rank, num_processes, split, d)
                    for d in dirs
                ]

        # Seeded RNG — same seed on all DDP ranks so source selection is consistent.
        # All ranks call next_batch() in lockstep, so the sequence of choices
        # is identical across ranks.
        self._rng = random.Random(42)

        # Expose primary-source attributes for backward compatibility with
        # train_gpt.py which reads/writes current_shard and current_position.
        self._primary = self._loaders[0]

    # ---- Backward-compatible attribute proxies --------------------------------

    @property
    def shards(self):
        return self._primary.shards

    @property
    def current_shard(self):
        return self._primary.current_shard

    @current_shard.setter
    def current_shard(self, value):
        self._primary.current_shard = value

    @property
    def current_position(self):
        return self._primary.current_position

    @current_position.setter
    def current_position(self, value):
        self._primary.current_position = value

    @property
    def tokens(self):
        return self._primary.tokens

    @tokens.setter
    def tokens(self, value):
        self._primary.tokens = value

    # ---- Core interface -------------------------------------------------------

    def reset(self):
        """Reset all sources to the beginning."""
        for loader in self._loaders:
            loader.reset()
        self._rng = random.Random(42)

    def next_batch(self):
        """Return the next (x, y) batch.

        In multi-source train mode, picks a source proportionally to its weight.
        In single-source or val mode, always reads from the single source.
        """
        if len(self._loaders) == 1:
            return self._loaders[0].next_batch()
        idx = self._rng.choices(range(len(self._loaders)),
                                weights=self._weights, k=1)[0]
        return self._loaders[idx].next_batch()

    # ---- State dict for checkpoint / resume ----------------------------------

    def state_dict(self) -> dict:
        """Return a dict capturing full loader state for resumption."""
        return {
            "multi":   self._multi,
            "sources": [l.state_dict() for l in self._loaders],
            "rng":     self._rng.getstate(),
        }

    def load_state_dict(self, state: dict):
        """Restore loader state from a checkpoint dict."""
        for loader, s in zip(self._loaders, state["sources"]):
            loader.load_state_dict(s)
        if "rng" in state:
            rng_state = state["rng"]
            if isinstance(rng_state, list):
                # JSON converts tuples to lists — reconstruct nested tuple
                # random.getstate() format: (version, internalstate_tuple, gauss_next)
                rng_state = (rng_state[0], tuple(rng_state[1]), rng_state[2])
            self._rng.setstate(rng_state)
