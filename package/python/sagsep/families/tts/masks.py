from __future__ import annotations

import numpy as np

from sagsep.types import mask_key_bytes, pack_masks

__all__ = [
    "pack_selection_rows",
    "popcount",
    "row_key",
    "first_seen_unique_rows",
]

_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")
if not _HAS_BITWISE_COUNT:
    _POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def pack_selection_rows(selections: np.ndarray) -> np.ndarray:
    return pack_masks(selections > 0.5)


def popcount(masks: np.ndarray) -> np.ndarray:
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(masks).sum(axis=-1, dtype=np.int64)
    as_u8 = masks.view(np.uint8)
    return _POP8[as_u8].sum(axis=-1, dtype=np.int64)


def row_key(mask_row: np.ndarray) -> bytes:
    return mask_key_bytes(mask_row)


def first_seen_unique_rows(masks: np.ndarray) -> np.ndarray:
    if masks.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    _, first = np.unique(masks, axis=0, return_index=True)
    return np.sort(first).astype(np.int64, copy=False)
