from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from sagsep.families.tts.config import TTSConfig
from sagsep.families.tts.masks import popcount
from sagsep.families.tts.proposal import Proposal

__all__ = ["FlowState", "flow_scale", "scaled_limit", "apply_cut_flow"]

_INITIAL_CAPACITY = 256


@dataclass(slots=True)
class FlowState:
    vehicle_count: int
    n_nodes: int
    mask_words: int
    count: int = 0
    emitted: int = 0
    seen_keys: set = field(default_factory=set)
    selected_sources: List[str] = field(default_factory=list)
    _masks: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        if self._masks is None:
            self._masks = np.zeros((_INITIAL_CAPACITY, self.mask_words), dtype=np.uint64)

    def masks(self) -> np.ndarray:
        return self._masks[: self.count]

    def add(self, item: Proposal) -> None:
        if self.count == self._masks.shape[0]:
            grown = np.zeros((self._masks.shape[0] * 2, self.mask_words), dtype=np.uint64)
            grown[: self.count] = self._masks[: self.count]
            self._masks = grown
        self._masks[self.count] = item.mask
        self.count += 1
        self.seen_keys.add(item.key)
        self.selected_sources.append(item.source)


def flow_scale(config: TTSConfig, vehicle_count: int) -> float:
    start = int(config.vehicle_tighten_start)
    if start <= 0 or vehicle_count <= start:
        return 1.0
    raw = (float(start) / float(max(vehicle_count, 1))) ** float(
        config.vehicle_tighten_power
    )
    return max(float(config.vehicle_tighten_min_scale), min(1.0, raw))


def scaled_limit(value: int, scale: float) -> int:
    if int(value) <= 0:
        return 0
    return max(1, int(math.ceil(float(value) * scale)))


def apply_cut_flow(
    proposals: List[Proposal],
    *,
    config: TTSConfig,
    flow_state: FlowState,
    diagnostics: Optional[Dict[str, object]] = None,
) -> List[Proposal]:
    if diagnostics is not None:
        diagnostics["near_duplicate_hit_count"] = 0
        diagnostics["near_duplicate_source_pairs"] = {}
    if not proposals:
        return []
    scale = flow_scale(config, flow_state.vehicle_count)
    vehicle_limit = scaled_limit(config.per_vehicle_quota, scale)
    global_limit = scaled_limit(config.global_quota, scale)
    source_limits = {
        str(source): scaled_limit(int(limit), scale)
        for source, limit in config.source_quotas
        if int(limit) > 0
    }
    n = flow_state.n_nodes
    hamming_limit = int(config.near_duplicate_hamming)

    remaining = [item for item in proposals if item.key not in flow_state.seen_keys]
    if not remaining:
        return []
    masks = np.stack([item.mask for item in remaining], axis=0)
    prior = flow_state.masks()
    if prior.shape[0]:
        distances_all = popcount(masks[:, None, :] ^ prior[None, :, :])
        min_hamming = distances_all.min(axis=1)
    else:
        min_hamming = np.full(len(remaining), n, dtype=np.int64)

    near_duplicate_hits = np.zeros(len(remaining), dtype=bool)
    blocker_sources = np.full(len(remaining), None, dtype=object)
    track_blockers = diagnostics is not None
    if hamming_limit > 0 and prior.shape[0]:
        if track_blockers:
            nearest = np.full(len(remaining), n, dtype=np.int64)
            for ref_idx in range(prior.shape[0]):
                distances = distances_all[:, ref_idx]
                closer = distances < nearest
                blocker_sources[closer] = flow_state.selected_sources[ref_idx]
                nearest = np.minimum(nearest, distances)
        near_duplicate_hits = min_hamming <= hamming_limit

    active = np.ones(len(remaining), dtype=bool)
    sources = np.asarray([item.source for item in remaining], dtype=object)
    base_scores = np.asarray([float(item.score) for item in remaining], dtype=np.float64)
    selected: List[Proposal] = []
    local_counts: Dict[str, int] = {}
    while np.any(active):
        if vehicle_limit > 0 and len(selected) >= vehicle_limit:
            break
        if global_limit > 0 and flow_state.emitted + len(selected) >= global_limit:
            break

        eligible = active.copy()
        for source_name, source_limit in source_limits.items():
            if local_counts.get(source_name, 0) >= source_limit:
                eligible &= sources != source_name
        if (flow_state.count > 0 or selected) and hamming_limit > 0:
            eligible &= min_hamming > hamming_limit
        eligible_idx = np.flatnonzero(eligible)
        if eligible_idx.size == 0:
            break
        diversity = min_hamming.astype(np.float64) / float(max(n - 1, 1))
        rank_scores = base_scores + float(config.diversity_weight) * diversity
        best_idx = int(eligible_idx[int(np.argmax(rank_scores[eligible_idx]))])
        chosen = remaining[best_idx]
        active[best_idx] = False
        selected.append(chosen)
        local_counts[chosen.source] = local_counts.get(chosen.source, 0) + 1
        distances = popcount(masks ^ masks[best_idx][None, :])
        if hamming_limit > 0 and track_blockers:
            newly_blocked = (
                active
                & ~near_duplicate_hits
                & (distances <= hamming_limit)
            )
            blocker_sources[newly_blocked] = chosen.source
            near_duplicate_hits |= newly_blocked
        min_hamming = np.minimum(min_hamming, distances)

    for item in selected:
        flow_state.add(item)
    flow_state.emitted += len(selected)
    if diagnostics is not None:
        source_pairs: Dict[str, int] = {}
        for index in np.flatnonzero(near_duplicate_hits).tolist():
            blocker = blocker_sources[index]
            pair = f"{remaining[index].source}->{blocker or 'unknown'}"
            source_pairs[pair] = source_pairs.get(pair, 0) + 1
        diagnostics["near_duplicate_hit_count"] = int(np.count_nonzero(near_duplicate_hits))
        diagnostics["near_duplicate_source_pairs"] = dict(sorted(source_pairs.items()))
    return selected
