# sphinx/utils/bits.py
import random
from typing import List

def rand_bits(rng: random.Random, n: int, min_ones=3, max_ones=None) -> List[int]:
    L = n*n
    if max_ones is None: max_ones = L - 3
    while True:
        bits = [1 if rng.random() < 0.5 else 0 for _ in range(L)]
        k = sum(bits)
        if min_ones <= k <= max_ones:
            return bits

def xor_bits(bits: List[int], mask_idx: List[int]) -> List[int]:
    out = bits[:]
    for i in mask_idx:
        if 0 <= i < len(out):
            out[i] ^= 1
    return out
