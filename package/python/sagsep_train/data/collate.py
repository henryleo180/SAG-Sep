from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from sagsep.features import GraphFeatures, build_graph_features, build_lp_support, vehicle_replica_count
from sagsep.types import Instance

__all__ = [
    "SUPPORT_RULE_THRESHOLDS",
    "TrainBatch",
    "state_instance",
    "state_features",
    "collate_states",
    "collator",
]

SUPPORT_RULE_THRESHOLDS: dict[str, float] = {
    "x_gt_0": 0.0,
    "x_ge_1e-6": 1e-6,
}


@dataclass(frozen=True)
class TrainBatch:
    edge_index: torch.Tensor
    node_attr: torch.Tensor
    edge_attr: torch.Tensor
    target: torch.Tensor
    depot_mask: torch.Tensor
    graph_id: torch.Tensor
    n_states: int
    n_replicas: int
    replicas_per_state: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.node_attr.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: Any, *, non_blocking: bool = False) -> "TrainBatch":
        return TrainBatch(
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            node_attr=self.node_attr.to(device, non_blocking=non_blocking),
            edge_attr=self.edge_attr.to(device, non_blocking=non_blocking),
            target=self.target.to(device, non_blocking=non_blocking),
            depot_mask=self.depot_mask.to(device, non_blocking=non_blocking),
            graph_id=self.graph_id.to(device, non_blocking=non_blocking),
            n_states=self.n_states,
            n_replicas=self.n_replicas,
            replicas_per_state=self.replicas_per_state.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "TrainBatch":
        return TrainBatch(
            edge_index=self.edge_index.pin_memory(),
            node_attr=self.node_attr.pin_memory(),
            edge_attr=self.edge_attr.pin_memory(),
            target=self.target.pin_memory(),
            depot_mask=self.depot_mask.pin_memory(),
            graph_id=self.graph_id.pin_memory(),
            n_states=self.n_states,
            n_replicas=self.n_replicas,
            replicas_per_state=self.replicas_per_state.pin_memory(),
        )


def _threshold_for(support_rule: str) -> float:
    try:
        return SUPPORT_RULE_THRESHOLDS[support_rule]
    except KeyError:
        raise ValueError(
            f"unknown support_rule {support_rule!r}, expected one of "
            f"{sorted(SUPPORT_RULE_THRESHOLDS)}"
        ) from None


def state_instance(state: Any) -> Instance:
    record = state.instance
    return Instance(
        name=str(record.id),
        n_nodes=int(record.n_nodes),
        depot=0,
        coords=np.asarray(record.coords, dtype=np.float64),
        demand=np.asarray(record.demand, dtype=np.int64),
        capacity=int(record.capacity),
        vehicles=int(record.k_fleet),
    )


def state_features(
    state: Any,
    *,
    support_rule: str = "x_gt_0",
    cross_depot: bool = True,
) -> tuple[GraphFeatures, np.ndarray]:
    threshold = _threshold_for(support_rule)
    instance = state_instance(state)

    s_hat = np.asarray(state.s_hat)
    if s_hat.ndim != 2 or s_hat.shape[0] != instance.n_nodes:
        raise ValueError(
            f"state {instance.name!r}: s_hat must be (n_nodes, k_label) = "
            f"({instance.n_nodes}, k), got {s_hat.shape}"
        )
    k_label = int(s_hat.shape[1])
    record_k = int(state.instance.k_label)
    replicas = vehicle_replica_count(instance)
    if not (k_label == record_k == replicas):
        raise ValueError(
            f"state {instance.name!r}: label width {k_label}, instance k_label "
            f"{record_k} and replica count {replicas} must agree"
        )

    edge_u = np.asarray(state.support_u, dtype=np.int64)
    edge_v = np.asarray(state.support_v, dtype=np.int64)
    edge_x = np.asarray(state.support_x, dtype=np.float64)
    lp = build_lp_support(edge_u, edge_v, edge_x, n_nodes=instance.n_nodes, threshold=threshold)
    features = build_graph_features(
        instance,
        lp,
        vehicle_count=k_label,
        support_edge_u=edge_u,
        support_edge_v=edge_v,
        support_edge_x=edge_x,
        cross_depot=cross_depot,
    )
    return features, s_hat


def collate_states(
    states: Sequence[Any],
    *,
    support_rule: str = "x_gt_0",
    cross_depot: bool = True,
) -> TrainBatch:
    if len(states) == 0:
        raise ValueError("collate_states: at least one state is required")

    edge_parts: list[np.ndarray] = []
    edge_attr_parts: list[np.ndarray] = []
    node_attr_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    depot_parts: list[np.ndarray] = []
    graph_id_parts: list[np.ndarray] = []
    replicas_per_state: list[int] = []

    node_offset = 0
    replica_offset = 0
    for state in states:
        features, s_hat = state_features(
            state, support_rule=support_rule, cross_depot=cross_depot
        )
        n = int(features.n_nodes)
        k = int(features.vehicle_count)
        n_edges = int(features.edge_index.shape[1])

        block_offsets = node_offset + n * np.arange(k, dtype=np.int64)
        expanded = features.edge_index[None, :, :] + block_offsets[:, None, None]
        edge_parts.append(np.ascontiguousarray(expanded.transpose(1, 0, 2).reshape(2, k * n_edges)))
        edge_attr_parts.append(np.tile(features.edge_attr, (k, 1)))
        node_attr_parts.append(features.node_attr.reshape(k * n, features.node_attr.shape[2]))

        target_parts.append(np.ascontiguousarray(s_hat.T).reshape(-1).astype(np.float32))
        depot = np.zeros(k * n, dtype=bool)
        depot[::n] = True
        depot_parts.append(depot)
        graph_id_parts.append(
            np.repeat(replica_offset + np.arange(k, dtype=np.int64), n)
        )

        replicas_per_state.append(k)
        node_offset += k * n
        replica_offset += k

    edge_index = np.concatenate(edge_parts, axis=1)
    edge_attr = np.concatenate(edge_attr_parts, axis=0).astype(np.float32, copy=False)
    node_attr = np.concatenate(node_attr_parts, axis=0).astype(np.float32, copy=False)
    target = np.concatenate(target_parts)
    depot_mask = np.concatenate(depot_parts)
    graph_id = np.concatenate(graph_id_parts)

    return TrainBatch(
        edge_index=torch.from_numpy(np.ascontiguousarray(edge_index)),
        node_attr=torch.from_numpy(np.ascontiguousarray(node_attr)),
        edge_attr=torch.from_numpy(np.ascontiguousarray(edge_attr)),
        target=torch.from_numpy(np.ascontiguousarray(target)),
        depot_mask=torch.from_numpy(np.ascontiguousarray(depot_mask)),
        graph_id=torch.from_numpy(np.ascontiguousarray(graph_id)),
        n_states=len(states),
        n_replicas=replica_offset,
        replicas_per_state=torch.from_numpy(
            np.asarray(replicas_per_state, dtype=np.int64)
        ),
    )


def collator(*, support_rule: str = "x_gt_0", cross_depot: bool = True):
    from functools import partial

    _threshold_for(support_rule)
    return partial(collate_states, support_rule=support_rule, cross_depot=cross_depot)
