from __future__ import annotations

import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from sagsep_train.data.format import InstanceRecord, ShardData, StateView, read_shard, read_shard_counts

__all__ = [
    "SHARD_SUFFIX",
    "shard_file_name",
    "ShardIndex",
    "detach_state",
    "TrainingDataset",
    "GlobalPermutationSampler",
    "ShardOrderSampler",
    "select_val_records",
    "write_record_ids",
    "read_record_ids",
    "make_dataloader",
    "worker_init_fn_for",
]

SHARD_SUFFIX = ".npz"


def shard_file_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("shard name must not be empty")
    if os.sep in text or (os.altsep and os.altsep in text) or "/" in text:
        raise ValueError(f"shard name must be a bare file name, got {text!r}")
    return text if text.endswith(SHARD_SUFFIX) else text + SHARD_SUFFIX


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


class ShardIndex:
    def __init__(
        self,
        dataset_dir: str | os.PathLike[str],
        shard_names: Sequence[str],
        *,
        cache_size: int = 2,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        names = [str(name) for name in shard_names]
        if not names:
            raise ValueError("ShardIndex: at least one shard is required")
        if int(cache_size) < 1:
            raise ValueError(f"ShardIndex: cache_size must be >= 1, got {cache_size}")
        self.cache_size = int(cache_size)

        self.shard_names: tuple[str, ...] = tuple(names)
        self.shard_paths: tuple[Path, ...] = tuple(
            self.dataset_dir / shard_file_name(name) for name in names
        )
        missing = [str(path) for path in self.shard_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("shard file(s) not found: " + ", ".join(missing))

        counts = [read_shard_counts(path) for path in self.shard_paths]
        self.instance_counts: tuple[int, ...] = tuple(int(pair[0]) for pair in counts)
        self.shard_sizes: tuple[int, ...] = tuple(int(pair[1]) for pair in counts)
        offsets = np.zeros(len(self.shard_sizes) + 1, dtype=np.int64)
        np.cumsum(np.asarray(self.shard_sizes, dtype=np.int64), out=offsets[1:])
        self._offsets = offsets
        self._n_states = int(offsets[-1])

        self._cache: "OrderedDict[int, ShardData]" = OrderedDict()
        self._decode_count = 0

    def __len__(self) -> int:
        return self._n_states

    @property
    def n_shards(self) -> int:
        return len(self.shard_names)

    @property
    def decode_count(self) -> int:
        return self._decode_count

    def locate(self, global_idx: int) -> tuple[int, int]:
        idx = int(global_idx)
        if idx < 0:
            idx += self._n_states
        if not 0 <= idx < self._n_states:
            raise IndexError(f"state index {global_idx} out of range (0 .. {self._n_states - 1})")
        shard_row = int(np.searchsorted(self._offsets, idx, side="right")) - 1
        return shard_row, idx - int(self._offsets[shard_row])

    def shard_bounds(self, shard_row: int) -> range:
        row = int(shard_row)
        if not 0 <= row < self.n_shards:
            raise IndexError(f"shard row {shard_row} out of range (0 .. {self.n_shards - 1})")
        return range(int(self._offsets[row]), int(self._offsets[row + 1]))

    def shard(self, shard_row: int) -> ShardData:
        row = int(shard_row)
        cached = self._cache.get(row)
        if cached is not None:
            self._cache.move_to_end(row)
            return cached
        data = read_shard(self.shard_paths[row])
        self._decode_count += 1
        self._cache[row] = data
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def state(self, global_idx: int) -> StateView:
        shard_row, state_row = self.locate(global_idx)
        return self.shard(shard_row).state(state_row)

    def instance_for(self, global_idx: int) -> InstanceRecord:
        shard_row, state_row = self.locate(global_idx)
        data = self.shard(shard_row)
        return data.instance(int(data.st_inst_row[state_row]))

    def clear_cache(self) -> None:
        self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"ShardIndex(dir={str(self.dataset_dir)!r}, shards={self.n_shards}, "
            f"states={self._n_states}, cache_size={self.cache_size})"
        )


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


def detach_state(state: StateView) -> StateView:
    inst = state.instance
    instance = InstanceRecord(
        id=inst.id,
        source_sha256=inst.source_sha256,
        n_nodes=inst.n_nodes,
        capacity=inst.capacity,
        k_fleet=inst.k_fleet,
        k_label=inst.k_label,
        termination=inst.termination,
        coords=np.array(inst.coords, copy=True),
        demand=np.array(inst.demand, copy=True),
    )
    return StateView(
        instance=instance,
        inst_row=state.inst_row,
        iteration=state.iteration,
        lp_value=state.lp_value,
        cuts_added=state.cuts_added,
        terminal=state.terminal,
        support_u=np.array(state.support_u, copy=True),
        support_v=np.array(state.support_v, copy=True),
        support_x=np.array(state.support_x, copy=True),
        s_hat=np.array(state.s_hat, copy=True),
        objective=np.array(state.objective, copy=True),
        rhs=np.array(state.rhs, copy=True),
        slot_violated=np.array(state.slot_violated, copy=True),
    )


class TrainingDataset(Dataset):
    def __init__(self, index: ShardIndex, record_ids: Sequence[int] | None = None) -> None:
        self.index = index
        if record_ids is None:
            self.record_ids: np.ndarray = np.arange(len(index), dtype=np.int64)
        else:
            ids = np.asarray(list(record_ids), dtype=np.int64)
            if ids.ndim != 1:
                raise ValueError("record_ids must be a 1-d sequence of state indices")
            if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= len(index)):
                raise IndexError(
                    f"record_ids out of range (0 .. {len(index) - 1}): "
                    f"[{int(ids.min())}, {int(ids.max())}]"
                )
            self.record_ids = ids

    def __len__(self) -> int:
        return int(self.record_ids.shape[0])

    def __getitem__(self, position: int) -> StateView:
        return detach_state(self.index.state(int(self.record_ids[int(position)])))

    def shard_sizes(self) -> list[int]:
        sizes = [0] * self.index.n_shards
        for record in self.record_ids.tolist():
            shard_row, _ = self.index.locate(int(record))
            sizes[shard_row] += 1
        return sizes


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------


class GlobalPermutationSampler(Sampler[int]):
    def __init__(self, n: int, *, seed: int, epoch: int = 0) -> None:
        self.n = int(n)
        if self.n < 0:
            raise ValueError(f"GlobalPermutationSampler: n must be >= 0, got {n}")
        self.seed = int(seed)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def permutation(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.n, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        return iter(self.permutation())

    def __len__(self) -> int:
        return self.n


class ShardOrderSampler(Sampler[int]):
    def __init__(self, shard_sizes: Sequence[int], *, seed: int, epoch: int = 0) -> None:
        sizes = [int(size) for size in shard_sizes]
        if any(size < 0 for size in sizes):
            raise ValueError("ShardOrderSampler: shard sizes must be non-negative")
        self.shard_sizes = sizes
        self.seed = int(seed)
        self.epoch = int(epoch)
        self._starts: list[int] = []
        running = 0
        for size in sizes:
            self._starts.append(running)
            running += size
        self.n = running

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def permutation(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order: list[int] = []
        shard_order = torch.randperm(len(self.shard_sizes), generator=generator).tolist()
        for shard_row in shard_order:
            size = self.shard_sizes[shard_row]
            if size == 0:
                continue
            start = self._starts[shard_row]
            inner = torch.randperm(size, generator=generator).tolist()
            order.extend(start + position for position in inner)
        return order

    def __iter__(self) -> Iterator[int]:
        return iter(self.permutation())

    def __len__(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# validation record ids
# ---------------------------------------------------------------------------


def select_val_records(n_total: int, val_limit: int | None, *, seed: int) -> list[int]:
    total = int(n_total)
    if total < 0:
        raise ValueError(f"select_val_records: n_total must be >= 0, got {n_total}")
    if val_limit is None or int(val_limit) >= total:
        return list(range(total))
    limit = int(val_limit)
    if limit <= 0:
        return []
    generator = torch.Generator().manual_seed(int(seed))
    chosen = torch.randperm(total, generator=generator)[:limit].tolist()
    chosen.sort()
    return [int(value) for value in chosen]


def write_record_ids(path: str | os.PathLike[str], ids: Sequence[int]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = [int(value) for value in ids]
    payload = {"count": len(values), "record_ids": values}
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
    return target


def read_record_ids(path: str | os.PathLike[str]) -> list[int]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        values = payload.get("record_ids")
        if values is None:
            raise ValueError(f"{path}: no 'record_ids' key in the record-id file")
    else:
        values = payload
    if not isinstance(values, list):
        raise ValueError(f"{path}: record ids must be a list, got {type(values).__name__}")
    return [int(value) for value in values]


# ---------------------------------------------------------------------------
# dataloader
# ---------------------------------------------------------------------------


def worker_init_fn_for(seed: int) -> Callable[[int], None]:
    base = int(seed)

    def init(worker_id: int) -> None:
        worker_seed = base + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)

    return init


def make_dataloader(
    dataset: TrainingDataset | ShardIndex,
    *,
    batch_size: int,
    sampler: Sampler[int] | Sequence[int],
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    collate_fn: Callable[[list[Any]], Any],
    seed: int = 0,
) -> DataLoader:
    if isinstance(dataset, ShardIndex):
        dataset = TrainingDataset(dataset)
    if int(batch_size) < 1:
        raise ValueError(f"make_dataloader: batch_size must be >= 1, got {batch_size}")
    if int(num_workers) < 0:
        raise ValueError(f"make_dataloader: num_workers must be >= 0, got {num_workers}")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=bool(drop_last),
        pin_memory=bool(pin_memory),
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn_for(seed),
        persistent_workers=False,
    )
