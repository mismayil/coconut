# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import List
import random, torch, os
import numpy as np
from dataclasses import dataclass

class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class ComputeRange:
    start: int
    end: int
    mode: str # "lang", "latent"

    def __contains__(self, idx: int) -> bool:
        return self.start <= idx < self.end

class BatchComputeRangeIterator:
    def __init__(self, batch_ranges: List[List[ComputeRange]]):
        self.batch_ranges = batch_ranges
        self.batch_last_ids = [0] * len(batch_ranges)
        self.batch_size = len(batch_ranges)

    def __iter__(self):
        return self

    def __next__(self):
        current_id = min(self.batch_last_ids)

        latent_ranges = []
        lang_ranges = []

        for idx, cranges in enumerate(self.batch_ranges):
            for crange in cranges:
                if current_id in crange and current_id >= self.batch_last_ids[idx]:
                    if crange.mode == "latent":
                        latent_ranges.append((idx, crange))
                    else:
                        lang_ranges.append((idx, crange))
        
        next_latent_range = (None, None)
        next_lang_range = (None, None)

        if not latent_ranges and not lang_ranges:
            raise StopIteration

        if latent_ranges:
            min_end = min([lr.end for _, lr in latent_ranges])
            next_latent_indices = [idx for idx, _ in latent_ranges]
            next_latent_range = (next_latent_indices, ComputeRange(current_id, min_end, "latent"))
            for idx in next_latent_indices:
                self.batch_last_ids[idx] = min_end

        if lang_ranges:
            min_end = min([lr.end for _, lr in lang_ranges])
            next_lang_indices = [idx for idx, _ in lang_ranges]
            next_lang_range = (next_lang_indices, ComputeRange(current_id, min_end, "lang"))
            for idx in next_lang_indices:
                self.batch_last_ids[idx] = min_end

        return next_lang_range, next_latent_range