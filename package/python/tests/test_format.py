from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from sagsep_train.data.format import (
    FEATURE_CONTRACT,
    SHARD_DTYPES,
    SHARD_KEYS,
    SHARD_SCHEMA_VERSION,
    InstanceRecord,
    ShardBuilder,
    ShardData,
    StateRecord,
    ceil_div,
    merge_shards,
    read_shard,
    read_shard_counts,
    sha256_file,
    write_shard,
)

# --------------------------------------------------------------------------
# synthetic shard helpers (shared with the other suites)
# --------------------------------------------------------------------------

DEMO_INSTANCES = (
    ("alpha", [7, 9, 6, 8], 10, 4, "closed", 3),
    ("bravo", [3, 2, 4, 1, 2], 15, 2, "iteration_cap", 2),
    ("charlie", [11, 9], 10, 2, "time_limit", 4),
)


def fake_sha256(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def stable_seed(*parts) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def make_instance(name: str, demand: list[int], capacity: int, k_fleet: int, termination: str) -> InstanceRecord:
    n_nodes = len(demand) + 1
    rng = np.random.default_rng(stable_seed(name))
    coords = rng.uniform(0.0, 100.0, size=(n_nodes, 2))
    demand_array = np.asarray([0] + list(demand), dtype=np.int64)
    return InstanceRecord(
        id=name,
        source_sha256=fake_sha256(name),
        n_nodes=n_nodes,
        capacity=capacity,
        k_fleet=k_fleet,
        k_label=ceil_div(int(demand_array.sum()), capacity),
        termination=termination,
        coords=coords,
        demand=demand_array,
    )


def make_state(rec: InstanceRecord, iteration: int, terminal: bool) -> StateRecord:
    rng = np.random.default_rng(stable_seed(rec.id, iteration))
    n_nodes = rec.n_nodes
    pairs = [(u, v) for u in range(n_nodes) for v in range(u + 1, n_nodes)]
    keep = rng.choice(len(pairs), size=max(2, len(pairs) // 2), replace=False)
    chosen = [pairs[i] for i in sorted(keep)]
    support_u = np.asarray([u for u, _ in chosen], dtype=np.int32)
    support_v = np.asarray([v for _, v in chosen], dtype=np.int32)
    support_x = rng.uniform(0.05, 1.0, size=len(chosen))

    s_hat = (rng.random((n_nodes, rec.k_label)) < 0.5).astype(np.uint8)
    s_hat[0, :] = 0
    return StateRecord(
        iteration=iteration,
        lp_value=100.0 + iteration,
        cuts_added=int(iteration * 2),
        terminal=terminal,
        support_u=support_u,
        support_v=support_v,
        support_x=support_x,
        s_hat=s_hat,
        objective=np.asarray(
            [1.5 + 0.25 * m + iteration for m in range(rec.k_label)], dtype=np.float64
        ),
        rhs=np.asarray([1 + m for m in range(rec.k_label)], dtype=np.int32),
        slot_violated=np.asarray([m % 2 for m in range(rec.k_label)], dtype=np.uint8),
    )


def make_demo_shard(instances=DEMO_INSTANCES) -> ShardData:
    builder = ShardBuilder()
    for name, demand, capacity, k_fleet, termination, n_states in instances:
        rec = make_instance(name, demand, capacity, k_fleet, termination)
        row = builder.add_instance(rec)
        for iteration in range(1, n_states + 1):
            builder.add_state(row, make_state(rec, iteration, terminal=iteration == n_states))
    return builder.build()


def single_instance_shards(data: ShardData) -> list[ShardData]:
    out: list[ShardData] = []
    for i in range(data.n_instances):
        builder = ShardBuilder()
        row = builder.add_instance(data.instance(i))
        for j in data.state_rows(i):
            builder.add_state(row, data.state_record(j))
        out.append(builder.build())
    return out


def concat_shards(shards) -> ShardData:
    builder = ShardBuilder()
    for shard in shards:
        for i in range(shard.n_instances):
            row = builder.add_instance(shard.instance(i))
            for j in shard.state_rows(i):
                builder.add_state(row, shard.state_record(j))
    return builder.build()


def assert_shards_equal(left: ShardData, right: ShardData) -> None:
    left_arrays = left.to_arrays()
    right_arrays = right.to_arrays()
    assert list(left_arrays) == list(SHARD_KEYS)
    assert list(right_arrays) == list(SHARD_KEYS)
    for key in SHARD_KEYS:
        a, b = left_arrays[key], right_arrays[key]
        assert a.dtype == b.dtype, f"{key}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{key}: shape {a.shape} != {b.shape}"
        assert np.array_equal(a, b), f"{key}: values differ"


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_builder_produces_declared_dtypes():
    data = make_demo_shard()
    arrays = data.to_arrays()
    assert list(arrays) == list(SHARD_KEYS)
    for key, expected in SHARD_DTYPES.items():
        found = arrays[key].dtype
        if expected == "U":
            assert found.kind == "U", key
        else:
            assert found == np.dtype(expected), key
    assert data.n_instances == 3
    assert data.n_states == 3 + 2 + 4
    assert [int(k) for k in data.inst_k_label] == [3, 1, 2]


def test_write_read_round_trip(tmp_path):
    data = make_demo_shard()
    path = tmp_path / "shard_0001.npz"
    write_shard(path, data)
    assert_shards_equal(read_shard(path), data)
    assert read_shard_counts(path) == (data.n_instances, data.n_states)


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    data = make_demo_shard()
    path = tmp_path / "shard_0001.npz"
    write_shard(path, data)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["shard_0001.npz"]
    write_shard(path, data)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["shard_0001.npz"]


def test_archive_has_no_object_arrays(tmp_path):
    path = tmp_path / "shard_0001.npz"
    write_shard(path, make_demo_shard())
    with np.load(path, allow_pickle=False) as archive:
        assert sorted(archive.files) == sorted(SHARD_KEYS)
        for key in archive.files:
            assert archive[key].dtype.kind in "iufU", f"{key}: dtype {archive[key].dtype}"


def test_instance_view_matches_source():
    data = make_demo_shard()
    for i, (name, demand, capacity, k_fleet, termination, _) in enumerate(DEMO_INSTANCES):
        rec = data.instance(i)
        assert rec.id == name
        assert rec.source_sha256 == fake_sha256(name)
        assert rec.n_nodes == len(demand) + 1
        assert rec.capacity == capacity
        assert rec.k_fleet == k_fleet
        assert rec.termination == termination
        assert rec.coords.shape == (rec.n_nodes, 2)
        assert np.array_equal(rec.demand, np.asarray([0] + demand, dtype=np.int64))
        assert rec.k_label == ceil_div(sum(demand), capacity)


def test_state_view_slices_are_correct():
    data = make_demo_shard()
    j = 0
    for i, (name, demand, capacity, k_fleet, termination, n_states) in enumerate(DEMO_INSTANCES):
        rec = make_instance(name, demand, capacity, k_fleet, termination)
        for iteration in range(1, n_states + 1):
            expected = make_state(rec, iteration, terminal=iteration == n_states)
            view = data.state(j)
            assert view.inst_row == i
            assert view.instance.id == name
            assert view.iteration == iteration
            assert view.lp_value == pytest.approx(expected.lp_value)
            assert view.cuts_added == expected.cuts_added
            assert view.terminal == (iteration == n_states)
            assert np.array_equal(view.support_u, expected.support_u)
            assert np.array_equal(view.support_v, expected.support_v)
            assert np.array_equal(view.support_x, expected.support_x)
            assert view.s_hat.shape == (rec.n_nodes, rec.k_label)
            assert np.array_equal(view.s_hat, expected.s_hat)
            assert np.array_equal(view.objective, expected.objective)
            assert np.array_equal(view.rhs, expected.rhs)
            assert np.array_equal(view.slot_violated, expected.slot_violated)
            j += 1
    assert j == data.n_states
    assert list(data.state_rows(0)) == [0, 1, 2]
    assert list(data.state_rows(1)) == [3, 4]
    assert list(data.state_rows(2)) == [5, 6, 7, 8]


def test_read_shard_refuses_other_schema_version(tmp_path):
    path = tmp_path / "shard_0001.npz"
    write_shard(path, make_demo_shard())
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["schema_version"] = np.asarray(SHARD_SCHEMA_VERSION + 1, dtype=np.int32)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="schema_version"):
        read_shard(path)


def test_read_shard_refuses_other_feature_contract(tmp_path):
    path = tmp_path / "shard_0001.npz"
    write_shard(path, make_demo_shard())
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["feature_contract"] = np.asarray("some_other_contract_v9", dtype=np.str_)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="feature_contract"):
        read_shard(path)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "payload.bin"
    payload = bytes(range(256)) * 17
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------


def test_merge_shards_rebases_offsets(tmp_path):
    original = make_demo_shard()
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    inputs = []
    for i, shard in enumerate(single_instance_shards(original)):
        part = parts_dir / f"part_{i}.npz"
        write_shard(part, shard)
        inputs.append(part)

    out_dir = tmp_path / "merged"
    written = merge_shards(inputs, out_dir, shard_size=2)
    assert [path.name for path in written] == ["shard_0001.npz", "shard_0002.npz"]
    assert sorted(item.name for item in out_dir.iterdir()) == ["shard_0001.npz", "shard_0002.npz"]

    first, second = (read_shard(path) for path in written)
    assert (first.n_instances, second.n_instances) == (2, 1)
    assert int(second.inst_node_offsets[0]) == 0
    assert int(second.sup_offsets[0]) == 0
    assert int(second.lab_offsets[0]) == 0
    assert int(second.m_offsets[0]) == 0
    assert int(second.inst_node_offsets[-1]) == second.inst_demand.shape[0]
    assert list(np.unique(second.st_inst_row)) == [0]

    assert_shards_equal(concat_shards([first, second]), original)


def test_merge_shards_with_multi_instance_inputs(tmp_path):
    original = make_demo_shard()
    source = tmp_path / "all.npz"
    write_shard(source, original)
    written = merge_shards([source], tmp_path / "out", shard_size=5)
    assert [path.name for path in written] == ["shard_0001.npz"]
    assert_shards_equal(read_shard(written[0]), original)


def test_merge_shards_prefix_and_empty_input(tmp_path):
    original = make_demo_shard()
    source = tmp_path / "all.npz"
    write_shard(source, original)
    written = merge_shards([source], tmp_path / "out", shard_size=1, prefix="chunk")
    assert [path.name for path in written] == [
        "chunk_0001.npz",
        "chunk_0002.npz",
        "chunk_0003.npz",
    ]
    assert merge_shards([], tmp_path / "empty", shard_size=4) == []


def test_merge_shards_rejects_bad_shard_size(tmp_path):
    with pytest.raises(ValueError, match="shard_size"):
        merge_shards([], tmp_path, shard_size=0)


# --------------------------------------------------------------------------
# builder validation
# --------------------------------------------------------------------------


def _instance() -> InstanceRecord:
    return make_instance("alpha", [7, 9, 6, 8], 10, 4, "closed")


def test_builder_rejects_bad_coords_shape():
    builder = ShardBuilder()
    rec = _instance()
    broken = replace(rec, coords=rec.coords[:, :1])
    with pytest.raises(ValueError, match=r"coords: expected shape"):
        builder.add_instance(broken)


def test_builder_rejects_wrong_k_label():
    builder = ShardBuilder()
    rec = _instance()
    broken = replace(rec, k_label=rec.k_label + 1)
    with pytest.raises(ValueError, match=r"k_label must equal ceil"):
        builder.add_instance(broken)


def test_builder_rejects_nonzero_depot_demand():
    builder = ShardBuilder()
    rec = _instance()
    demand = rec.demand.copy()
    demand[0] = 5
    broken = replace(rec, demand=demand)
    with pytest.raises(ValueError, match="depot demand must be 0"):
        builder.add_instance(broken)


def test_builder_rejects_negative_demand():
    builder = ShardBuilder()
    rec = _instance()
    demand = rec.demand.copy()
    demand[1] = -1
    broken = replace(rec, demand=demand)
    with pytest.raises(ValueError, match="non-negative"):
        builder.add_instance(broken)


def test_builder_rejects_bad_termination():
    builder = ShardBuilder()
    rec = _instance()
    broken = replace(rec, termination="exploded")
    with pytest.raises(ValueError, match="termination must be one of"):
        builder.add_instance(broken)


def test_builder_rejects_bad_source_sha():
    builder = ShardBuilder()
    rec = _instance()
    broken = replace(rec, source_sha256="not-a-digest")
    with pytest.raises(ValueError, match="source_sha256"):
        builder.add_instance(broken)


def test_builder_rejects_duplicate_instance_id():
    builder = ShardBuilder()
    builder.add_instance(_instance())
    with pytest.raises(ValueError, match="duplicate instance id"):
        builder.add_instance(_instance())


def test_builder_rejects_float_demand():
    builder = ShardBuilder()
    rec = _instance()
    broken = replace(rec, demand=rec.demand.astype(np.float64))
    with pytest.raises(ValueError, match="expected integer values"):
        builder.add_instance(broken)


def test_builder_rejects_bad_s_hat_shape():
    builder = ShardBuilder()
    rec = _instance()
    row = builder.add_instance(rec)
    state = make_state(rec, 1, terminal=True)
    broken = replace(state, s_hat=state.s_hat[:, :1])
    with pytest.raises(ValueError, match=r"s_hat: expected shape"):
        builder.add_state(row, broken)


def test_builder_rejects_unordered_support():
    builder = ShardBuilder()
    rec = _instance()
    row = builder.add_instance(rec)
    state = make_state(rec, 1, terminal=True)
    u = state.support_u.copy()
    v = state.support_v.copy()
    u[0], v[0] = v[0], u[0]
    broken = replace(state, support_u=u, support_v=v)
    with pytest.raises(ValueError, match="u < v"):
        builder.add_state(row, broken)


def test_builder_rejects_nonpositive_support_x():
    builder = ShardBuilder()
    rec = _instance()
    row = builder.add_instance(rec)
    state = make_state(rec, 1, terminal=True)
    x = state.support_x.copy()
    x[0] = 0.0
    broken = replace(state, support_x=x)
    with pytest.raises(ValueError, match="x > 0"):
        builder.add_state(row, broken)


def test_builder_rejects_nonzero_depot_label_row():
    builder = ShardBuilder()
    rec = _instance()
    row = builder.add_instance(rec)
    state = make_state(rec, 1, terminal=True)
    s_hat = state.s_hat.copy()
    s_hat[0, 0] = 1
    broken = replace(state, s_hat=s_hat)
    with pytest.raises(ValueError, match="depot row"):
        builder.add_state(row, broken)


def test_builder_rejects_iteration_gap():
    builder = ShardBuilder()
    rec = _instance()
    row = builder.add_instance(rec)
    with pytest.raises(ValueError, match="iteration must be 1"):
        builder.add_state(row, make_state(rec, 2, terminal=True))


def test_builder_rejects_state_after_terminal():
    builder = ShardBuilder()
    rec = _instance()
    row = builder.add_instance(rec)
    builder.add_state(row, make_state(rec, 1, terminal=True))
    with pytest.raises(ValueError, match="terminal"):
        builder.add_state(row, make_state(rec, 2, terminal=True))


def test_builder_rejects_unknown_instance_row():
    builder = ShardBuilder()
    rec = _instance()
    builder.add_instance(rec)
    with pytest.raises(ValueError, match="inst_row"):
        builder.add_state(7, make_state(rec, 1, terminal=True))


def test_build_requires_terminal_state():
    builder = ShardBuilder()
    rec = _instance()
    row = builder.add_instance(rec)
    builder.add_state(row, make_state(rec, 1, terminal=False))
    with pytest.raises(ValueError, match="no terminal state"):
        builder.build()


def test_build_requires_states_and_instances():
    with pytest.raises(ValueError, match="at least one instance"):
        ShardBuilder().build()
    builder = ShardBuilder()
    builder.add_instance(_instance())
    with pytest.raises(ValueError, match="has no state"):
        builder.build()


def test_builder_copies_its_input():
    builder = ShardBuilder()
    rec = _instance()
    row = builder.add_instance(rec)
    state = make_state(rec, 1, terminal=True)
    builder.add_state(row, state)
    rec.demand[1] = 999
    state.support_x[0] = -1.0
    data = builder.build()
    assert int(data.inst_demand[1]) == 7
    assert float(data.sup_x[0]) > 0.0
