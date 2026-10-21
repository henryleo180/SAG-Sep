from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sagsep_train.data.collate import collator
from sagsep_train.data.dataset import (
    GlobalPermutationSampler,
    ShardIndex,
    ShardOrderSampler,
    TrainingDataset,
    detach_state,
    make_dataloader,
    read_record_ids,
    select_val_records,
    shard_file_name,
    write_record_ids,
)
from sagsep_train.data.format import (
    InstanceRecord,
    ShardBuilder,
    StateRecord,
    ceil_div,
    write_shard,
)

N_NODES = 7
CAPACITY = 10
SHARD_LAYOUT = ((2, 3), (1, 2), (3, 2))


def _instance(name: str, seed: int) -> InstanceRecord:
    rng = np.random.default_rng(seed)
    coords = rng.random((N_NODES, 2)) * 100.0
    demand = np.zeros(N_NODES, dtype=np.int64)
    demand[1:] = rng.integers(1, 6, size=N_NODES - 1)
    return InstanceRecord(
        id=name,
        source_sha256=hashlib.sha256(name.encode("utf-8")).hexdigest(),
        n_nodes=N_NODES,
        capacity=CAPACITY,
        k_fleet=3,
        k_label=ceil_div(int(demand.sum()), CAPACITY),
        termination="closed",
        coords=coords,
        demand=demand,
    )


def _state(rec: InstanceRecord, iteration: int, terminal: bool, seed: int) -> StateRecord:
    rng = np.random.default_rng(seed)
    pairs = [(u, v) for u in range(rec.n_nodes) for v in range(u + 1, rec.n_nodes)]
    u = np.asarray([pair[0] for pair in pairs], dtype=np.int32)
    v = np.asarray([pair[1] for pair in pairs], dtype=np.int32)
    x = rng.random(len(pairs)) * 0.9 + 0.1
    s_hat = rng.integers(0, 2, size=(rec.n_nodes, rec.k_label)).astype(np.uint8)
    s_hat[0, :] = 0
    return StateRecord(
        iteration=iteration,
        lp_value=100.0 + iteration,
        cuts_added=iteration,
        terminal=terminal,
        support_u=u,
        support_v=v,
        support_x=x,
        s_hat=s_hat,
        objective=rng.random(rec.k_label),
        rhs=np.ones(rec.k_label, dtype=np.int32),
        slot_violated=np.zeros(rec.k_label, dtype=np.uint8),
    )


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("shards")
    seed = 0
    for shard_row, (n_instances, n_states) in enumerate(SHARD_LAYOUT, start=1):
        builder = ShardBuilder()
        for i in range(n_instances):
            seed += 1
            rec = _instance(f"inst-{shard_row}-{i}", seed)
            row = builder.add_instance(rec)
            for iteration in range(1, n_states + 1):
                builder.add_state(
                    row, _state(rec, iteration, iteration == n_states, seed * 100 + iteration)
                )
        write_shard(root / f"shard_{shard_row:04d}.npz", builder.build())
    return root


SHARD_NAMES = ["shard_0001", "shard_0002", "shard_0003"]
EXPECTED_SIZES = tuple(n_inst * n_states for n_inst, n_states in SHARD_LAYOUT)
TOTAL_STATES = sum(EXPECTED_SIZES)


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def test_shard_file_name() -> None:
    assert shard_file_name("shard_0001") == "shard_0001.npz"
    assert shard_file_name("shard_0001.npz") == "shard_0001.npz"
    with pytest.raises(ValueError):
        shard_file_name("sub/shard_0001")
    with pytest.raises(ValueError):
        shard_file_name("  ")


def test_index_lengths_and_order(dataset_dir: Path) -> None:
    index = ShardIndex(dataset_dir, SHARD_NAMES)
    assert index.n_shards == 3
    assert index.shard_sizes == EXPECTED_SIZES
    assert len(index) == TOTAL_STATES

    seen = [(index.instance_for(i).id, index.state(i).iteration) for i in range(len(index))]
    assert seen == [
        ("inst-1-0", 1), ("inst-1-0", 2), ("inst-1-0", 3),
        ("inst-1-1", 1), ("inst-1-1", 2), ("inst-1-1", 3),
        ("inst-2-0", 1), ("inst-2-0", 2),
        ("inst-3-0", 1), ("inst-3-0", 2),
        ("inst-3-1", 1), ("inst-3-1", 2),
        ("inst-3-2", 1), ("inst-3-2", 2),
    ]
    assert [index.locate(i) for i in (0, 5, 6, 7, 8, TOTAL_STATES - 1)] == [
        (0, 0), (0, 5), (1, 0), (1, 1), (2, 0), (2, 5)
    ]
    assert list(index.shard_bounds(1)) == [6, 7]


def test_index_accepts_names_with_and_without_the_suffix(dataset_dir: Path) -> None:
    index = ShardIndex(dataset_dir, ["shard_0001.npz", "shard_0002"])
    assert len(index) == EXPECTED_SIZES[0] + EXPECTED_SIZES[1]


def test_index_out_of_range_and_missing_shard(dataset_dir: Path) -> None:
    index = ShardIndex(dataset_dir, SHARD_NAMES)
    with pytest.raises(IndexError):
        index.state(TOTAL_STATES)
    assert index.state(-1).iteration == 2
    with pytest.raises(FileNotFoundError):
        ShardIndex(dataset_dir, ["shard_0009"])
    with pytest.raises(ValueError):
        ShardIndex(dataset_dir, [])


def test_lru_decodes_each_shard_once_in_a_sequential_pass(dataset_dir: Path) -> None:
    index = ShardIndex(dataset_dir, SHARD_NAMES, cache_size=1)
    for i in range(len(index)):
        index.state(i)
    assert index.decode_count == 3


def test_lru_evicts_least_recently_used(dataset_dir: Path) -> None:
    index = ShardIndex(dataset_dir, SHARD_NAMES, cache_size=1)
    index.state(0)
    assert index.decode_count == 1
    index.state(1)
    assert index.decode_count == 1
    index.state(6)
    index.state(0)
    assert index.decode_count == 3

    roomy = ShardIndex(dataset_dir, SHARD_NAMES, cache_size=2)
    for global_idx in (0, 6, 0, 6, 1, 7):
        roomy.state(global_idx)
    assert roomy.decode_count == 2

    roomy.clear_cache()
    roomy.state(0)
    assert roomy.decode_count == 3


def test_detach_state_owns_its_arrays(dataset_dir: Path) -> None:
    index = ShardIndex(dataset_dir, SHARD_NAMES)
    view = index.state(3)
    detached = detach_state(view)
    assert detached.support_x.base is None
    assert detached.instance.demand.base is None
    np.testing.assert_array_equal(detached.support_x, view.support_x)
    np.testing.assert_array_equal(detached.s_hat, view.s_hat)
    assert detached.instance.id == view.instance.id
    assert (detached.iteration, detached.terminal) == (view.iteration, view.terminal)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


def test_training_dataset_full_and_subset(dataset_dir: Path) -> None:
    index = ShardIndex(dataset_dir, SHARD_NAMES)
    full = TrainingDataset(index)
    assert len(full) == TOTAL_STATES
    assert full[2].iteration == 3

    subset = TrainingDataset(index, [1, 6, 13])
    assert len(subset) == 3
    assert [subset[i].instance.id for i in range(3)] == ["inst-1-0", "inst-2-0", "inst-3-2"]
    assert subset.shard_sizes() == [1, 1, 1]
    assert full.shard_sizes() == list(EXPECTED_SIZES)

    with pytest.raises(IndexError):
        TrainingDataset(index, [TOTAL_STATES])


# ---------------------------------------------------------------------------
# validation split
# ---------------------------------------------------------------------------


def test_select_val_records_matches_the_legacy_rule() -> None:
    n_total, limit, seed = 200, 32, 43
    chosen = select_val_records(n_total, limit, seed=seed)
    expected = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))[:limit].tolist()
    expected.sort()
    assert chosen == expected
    assert chosen == sorted(set(chosen))
    assert len(chosen) == limit


def test_select_val_records_edges() -> None:
    assert select_val_records(5, None, seed=1) == [0, 1, 2, 3, 4]
    assert select_val_records(5, 5, seed=1) == [0, 1, 2, 3, 4]
    assert select_val_records(5, 9, seed=1) == [0, 1, 2, 3, 4]
    assert select_val_records(5, 0, seed=1) == []
    assert select_val_records(0, 3, seed=1) == []
    assert select_val_records(50, 8, seed=43) == select_val_records(50, 8, seed=43)
    assert select_val_records(50, 8, seed=43) != select_val_records(50, 8, seed=44)


def test_record_ids_round_trip(tmp_path: Path) -> None:
    ids = select_val_records(100, 7, seed=43)
    path = write_record_ids(tmp_path / "ids" / "val_record_ids.json", ids)
    assert read_record_ids(path) == ids
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["count"] == len(ids)

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(ids), encoding="utf-8")
    assert read_record_ids(bare) == ids


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------


def test_global_permutation_sampler() -> None:
    n = 64
    sampler = GlobalPermutationSampler(n, seed=7)
    first = list(sampler)
    assert sorted(first) == list(range(n))
    assert len(sampler) == n

    sampler.set_epoch(1)
    second = list(sampler)
    assert sorted(second) == list(range(n))
    assert second != first

    again = GlobalPermutationSampler(n, seed=7, epoch=1)
    assert list(again) == second
    assert list(GlobalPermutationSampler(n, seed=8, epoch=1)) != second


def test_shard_order_sampler_is_shard_major() -> None:
    sizes = [3, 2, 4]
    sampler = ShardOrderSampler(sizes, seed=5)
    order = list(sampler)
    assert sorted(order) == list(range(sum(sizes)))
    assert len(sampler) == sum(sizes)

    blocks = {0: set(range(0, 3)), 1: set(range(3, 5)), 2: set(range(5, 9))}
    runs: list[int] = []
    for position in order:
        shard = next(row for row, members in blocks.items() if position in members)
        if not runs or runs[-1] != shard:
            runs.append(shard)
    assert sorted(runs) == [0, 1, 2]

    sampler.set_epoch(3)
    assert list(sampler) == list(ShardOrderSampler(sizes, seed=5, epoch=3))
    assert list(sampler) != order


def test_shard_order_sampler_skips_empty_shards() -> None:
    assert sorted(ShardOrderSampler([2, 0, 1], seed=1)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# dataloader
# ---------------------------------------------------------------------------


def _batches(loader) -> list[dict[str, torch.Tensor]]:
    out = []
    for batch in loader:
        out.append(
            {
                "edge_index": batch.edge_index,
                "node_attr": batch.node_attr,
                "edge_attr": batch.edge_attr,
                "target": batch.target,
                "depot_mask": batch.depot_mask,
                "graph_id": batch.graph_id,
                "replicas_per_state": batch.replicas_per_state,
            }
        )
    return out


def test_dataloader_is_worker_count_independent(dataset_dir: Path) -> None:
    index = ShardIndex(dataset_dir, SHARD_NAMES)
    dataset = TrainingDataset(index)
    collate_fn = collator(support_rule="x_gt_0", cross_depot=True)

    def build(num_workers: int):
        return make_dataloader(
            dataset,
            batch_size=4,
            sampler=GlobalPermutationSampler(len(dataset), seed=11, epoch=0),
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
            collate_fn=collate_fn,
            seed=11,
        )

    serial = _batches(build(0))
    parallel = _batches(build(2))
    assert len(serial) == 4
    assert len(serial) == len(parallel)
    for left, right in zip(serial, parallel):
        assert set(left) == set(right)
        for key in left:
            assert torch.equal(left[key], right[key]), key


def test_dataloader_drop_last_and_shard_index_argument(dataset_dir: Path) -> None:
    index = ShardIndex(dataset_dir, SHARD_NAMES)
    loader = make_dataloader(
        index,
        batch_size=4,
        sampler=GlobalPermutationSampler(len(index), seed=3),
        drop_last=True,
        collate_fn=collator(),
        seed=3,
    )
    batches = list(loader)
    assert len(batches) == 3
    assert all(batch.n_states == 4 for batch in batches)

    with pytest.raises(ValueError):
        make_dataloader(index, batch_size=0, sampler=[0], collate_fn=collator())


def test_validation_loader_reads_the_persisted_records(dataset_dir: Path, tmp_path: Path) -> None:
    index = ShardIndex(dataset_dir, SHARD_NAMES)
    ids = select_val_records(len(index), 6, seed=43)
    path = write_record_ids(tmp_path / "val_record_ids.json", ids)
    dataset = TrainingDataset(index, read_record_ids(path))
    loader = make_dataloader(
        dataset,
        batch_size=3,
        sampler=list(range(len(dataset))),
        collate_fn=collator(),
        seed=0,
    )
    batches = list(loader)
    assert sum(batch.n_states for batch in batches) == len(ids)
