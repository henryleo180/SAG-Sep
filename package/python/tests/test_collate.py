from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from sagsep_train.data.collate import collate_states, state_features
from sagsep_train.data.format import InstanceRecord, StateView

N_STATES = 8
N_NODES = 101
HOP_SEGMENT = {1.0: "support arc", 2.0: "two-hop shortcut", 0.0: "depot fill"}


def _fixture_path() -> Path:
    raw = os.environ.get("SAGSEP_LEGACY_FIXTURE", "").strip()
    if not raw:
        pytest.skip("SAGSEP_LEGACY_FIXTURE is not set (reference collate fixture)")
    path = Path(raw)
    if not path.is_file():
        pytest.skip(f"reference collate fixture not found: {path}")
    return path


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    with np.load(_fixture_path(), allow_pickle=False) as archive:
        wanted = [key for key in archive.files if key.startswith(("state", "batch/"))]
        return {key: archive[key] for key in wanted}


def _state_view(reference: dict[str, np.ndarray], i: int) -> StateView:
    demand = np.asarray(reference[f"state{i}/demand"], dtype=np.int64)
    capacity = int(reference[f"state{i}/capacity"])
    s_hat = np.asarray(reference[f"state{i}/s_hat"], dtype=np.uint8)
    k_label = int(s_hat.shape[1])
    instance = InstanceRecord(
        id=f"state{i}",
        source_sha256="0" * 64,
        n_nodes=int(demand.shape[0]),
        capacity=capacity,
        k_fleet=k_label,
        k_label=k_label,
        termination="closed",
        coords=np.zeros((int(demand.shape[0]), 2), dtype=np.float64),
        demand=demand,
    )
    return StateView(
        instance=instance,
        inst_row=i,
        iteration=int(reference[f"state{i}/iteration"]),
        lp_value=float(reference[f"state{i}/lp_value"]),
        cuts_added=0,
        terminal=False,
        support_u=np.asarray(reference[f"state{i}/support_u"], dtype=np.int32),
        support_v=np.asarray(reference[f"state{i}/support_v"], dtype=np.int32),
        support_x=np.asarray(reference[f"state{i}/support_x"], dtype=np.float64),
        s_hat=s_hat,
        objective=np.zeros(k_label, dtype=np.float64),
        rhs=np.ones(k_label, dtype=np.int32),
        slot_violated=np.zeros(k_label, dtype=np.uint8),
    )


@pytest.fixture(scope="module")
def states(reference: dict[str, np.ndarray]) -> list[StateView]:
    return [_state_view(reference, i) for i in range(N_STATES)]


@pytest.fixture(scope="module")
def batch(states: list[StateView]):
    return collate_states(states, support_rule="x_gt_0", cross_depot=True)


def _edge_report(
    column: int,
    ours: np.ndarray,
    theirs: np.ndarray,
    our_attr: np.ndarray,
    their_attr: np.ndarray,
) -> str:
    def describe(attr_row: np.ndarray) -> str:
        hop = float(attr_row[1])
        return f"{HOP_SEGMENT.get(hop, f'hop={hop}')} (value={float(attr_row[0]):.9g})"

    return (
        f"first mismatch at edge column {column}: "
        f"ours ({int(ours[0, column])} -> {int(ours[1, column])}) in the {describe(our_attr[column])} segment, "
        f"reference ({int(theirs[0, column])} -> {int(theirs[1, column])}) in the "
        f"{describe(their_attr[column])} segment"
    )


# ---------------------------------------------------------------------------
# ordered fidelity
# ---------------------------------------------------------------------------


def test_batch_shapes_match_the_reference(reference, batch) -> None:
    assert batch.n_states == N_STATES
    counts = reference["batch/vehicle_counts_per_instance"]
    assert batch.n_replicas == int(counts.sum())
    assert batch.num_nodes == int(reference["batch/nf"].shape[0])
    assert batch.num_edges == int(reference["batch/edge_index"].shape[1])


def test_edge_index_is_ordered_equal(reference, batch) -> None:
    ours = batch.edge_index.numpy()
    theirs = np.asarray(reference["batch/edge_index"], dtype=np.int64)
    assert ours.dtype == np.int64
    assert ours.shape == theirs.shape
    if not np.array_equal(ours, theirs):
        column = int(np.flatnonzero((ours != theirs).any(axis=0))[0])
        pytest.fail(
            _edge_report(
                column,
                ours,
                theirs,
                batch.edge_attr.numpy(),
                np.asarray(reference["batch/ef"], dtype=np.float32),
            )
        )


def test_edge_attr_is_bytewise_equal(reference, batch) -> None:
    ours = batch.edge_attr.numpy()
    theirs = np.asarray(reference["batch/ef"], dtype=np.float32)
    assert ours.dtype == np.float32 and ours.shape == theirs.shape
    if ours.tobytes() != theirs.tobytes():
        rows = np.flatnonzero((ours != theirs).any(axis=1))
        row = int(rows[0])
        column = int(np.flatnonzero(ours[row] != theirs[row])[0])
        name = "value" if column == 0 else "hop"
        pytest.fail(
            f"edge_attr differs in {len(rows)} row(s); first at row {row}, "
            f"{name} column: ours {ours[row]} vs reference {theirs[row]} "
            f"({_edge_report(row, batch.edge_index.numpy(), np.asarray(reference['batch/edge_index']), ours, theirs)})"
        )


def test_node_attr_is_bytewise_equal(reference, batch) -> None:
    ours = batch.node_attr.numpy()
    theirs = np.asarray(reference["batch/nf"], dtype=np.float32)
    assert ours.dtype == np.float32 and ours.shape == theirs.shape
    if ours.tobytes() != theirs.tobytes():
        rows = np.flatnonzero((ours != theirs).any(axis=1))
        row = int(rows[0])
        column = int(np.flatnonzero(ours[row] != theirs[row])[0])
        name = "demand/Q" if column == 0 else "band scalar m/max(K-1,1)"
        pytest.fail(
            f"node_attr differs in {len(rows)} row(s); first at row {row} "
            f"(replica {row // N_NODES}, node {row % N_NODES}), {name} column: "
            f"ours {ours[row]} vs reference {theirs[row]}"
        )


def test_targets_and_masks_are_ordered_equal(reference, batch) -> None:
    target = batch.target.numpy()
    reference_target = np.asarray(reference["batch/s_hat"], dtype=np.float32)
    assert target.dtype == np.float32
    np.testing.assert_array_equal(target, reference_target)

    depot = batch.depot_mask.numpy()
    assert depot.dtype == np.bool_
    np.testing.assert_array_equal(
        depot.astype(np.uint8), np.asarray(reference["batch/is_depot"], dtype=np.uint8)
    )
    assert depot.sum() == batch.n_replicas
    assert float(target[depot].max()) == 0.0


def test_graph_id_and_replica_counts_are_ordered_equal(reference, batch) -> None:
    np.testing.assert_array_equal(
        batch.graph_id.numpy(), np.asarray(reference["batch/batch"], dtype=np.int64)
    )
    counts = np.asarray(reference["batch/vehicle_counts_per_instance"], dtype=np.int64)
    np.testing.assert_array_equal(batch.replicas_per_state.numpy(), counts)
    assert list(counts) == [3, 4, 5, 6, 31, 32, 33, 34]
    assert batch.graph_id[0].item() == 0
    assert batch.graph_id[-1].item() == batch.n_replicas - 1


def test_state_features_agree_with_the_batch_blocks(reference, states, batch) -> None:
    node_offset = 0
    edge_offset = 0
    for i, state in enumerate(states):
        features, s_hat = state_features(state, support_rule="x_gt_0", cross_depot=True)
        k = features.vehicle_count
        n = features.n_nodes
        n_edges = features.edge_count
        assert (k, n) == (int(state.s_hat.shape[1]), N_NODES)

        block = batch.node_attr.numpy()[node_offset : node_offset + k * n]
        np.testing.assert_array_equal(block, features.node_attr.reshape(k * n, 2))

        first = batch.edge_index.numpy()[:, edge_offset : edge_offset + n_edges]
        np.testing.assert_array_equal(first, features.edge_index + node_offset)

        labels = batch.target.numpy()[node_offset : node_offset + k * n]
        np.testing.assert_array_equal(labels, np.ascontiguousarray(s_hat.T).reshape(-1))

        node_offset += k * n
        edge_offset += k * n_edges
    assert node_offset == batch.num_nodes
    assert edge_offset == batch.num_edges


def test_state_order_is_the_given_order(reference, states) -> None:
    forward = collate_states(states[:2], support_rule="x_gt_0")
    backward = collate_states(states[1::-1], support_rule="x_gt_0")
    counts_forward = forward.replicas_per_state.tolist()
    assert backward.replicas_per_state.tolist() == counts_forward[::-1]
    n_first = counts_forward[0] * N_NODES
    np.testing.assert_array_equal(
        forward.node_attr.numpy()[:n_first],
        backward.node_attr.numpy()[backward.num_nodes - n_first :],
    )


# ---------------------------------------------------------------------------
# support rules (synthetic, no fixture needed)
# ---------------------------------------------------------------------------


def _tiny_state(x_values: list[float]) -> StateView:
    n = 4
    demand = np.array([0, 4, 4, 4], dtype=np.int64)
    instance = InstanceRecord(
        id="tiny",
        source_sha256="0" * 64,
        n_nodes=n,
        capacity=10,
        k_fleet=2,
        k_label=2,
        termination="closed",
        coords=np.zeros((n, 2), dtype=np.float64),
        demand=demand,
    )
    s_hat = np.zeros((n, 2), dtype=np.uint8)
    s_hat[1:, 0] = 1
    return StateView(
        instance=instance,
        inst_row=0,
        iteration=1,
        lp_value=1.0,
        cuts_added=0,
        terminal=True,
        support_u=np.array([0, 0, 1, 2], dtype=np.int32),
        support_v=np.array([1, 3, 2, 3], dtype=np.int32),
        support_x=np.asarray(x_values, dtype=np.float64),
        s_hat=s_hat,
        objective=np.zeros(2, dtype=np.float64),
        rhs=np.ones(2, dtype=np.int32),
        slot_violated=np.zeros(2, dtype=np.uint8),
    )


def test_support_rule_threshold_drops_a_tiny_edge() -> None:
    state = _tiny_state([1.0, 5e-7, 1.0, 1.0])

    kept, _ = state_features(state, support_rule="x_gt_0")
    dropped, _ = state_features(state, support_rule="x_ge_1e-6")

    def support_pairs(features) -> set[tuple[int, int]]:
        index = features.edge_index
        hop = features.edge_attr[:, 1]
        return {
            (int(index[0, j]), int(index[1, j]))
            for j in range(index.shape[1])
            if float(hop[j]) == 1.0
        }

    assert (0, 3) in support_pairs(kept)
    assert (3, 0) in support_pairs(kept)
    assert (0, 3) not in support_pairs(dropped)
    assert (3, 0) not in support_pairs(dropped)
    assert {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)} <= support_pairs(dropped)


def test_unknown_support_rule_raises() -> None:
    with pytest.raises(ValueError, match="unknown support_rule"):
        state_features(_tiny_state([1.0, 1.0, 1.0, 1.0]), support_rule="x_gt_1")


def test_label_width_must_match_the_replica_count() -> None:
    state = _tiny_state([1.0, 1.0, 1.0, 1.0])
    broken = StateView(**{**state.__dict__, "s_hat": np.zeros((4, 3), dtype=np.uint8)})
    with pytest.raises(ValueError, match="must agree"):
        state_features(broken)


def test_collate_states_needs_at_least_one_state() -> None:
    with pytest.raises(ValueError, match="at least one state"):
        collate_states([])


def test_tiny_batch_is_consistent() -> None:
    batch = collate_states([_tiny_state([1.0, 0.5, 1.0, 1.0])] * 2)
    assert batch.n_states == 2
    assert batch.n_replicas == 4
    assert batch.replicas_per_state.tolist() == [2, 2]
    assert batch.num_nodes == 16
    assert batch.depot_mask.sum().item() == 4
    assert batch.graph_id.tolist() == sum(([g] * 4 for g in range(4)), [])
    assert batch.edge_index.max().item() == batch.num_nodes - 1
    assert batch.edge_index.dtype == torch.int64
    assert batch.node_attr.dtype == torch.float32
    assert batch.edge_attr.dtype == torch.float32
    assert batch.target.dtype == torch.float32
