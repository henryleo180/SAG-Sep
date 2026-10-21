from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
from sagsep import features
from sagsep.config import PRESET_SIZES, SagSepConfig, preset
from sagsep.loop import LearnedSeparator
from sagsep.model import load_policy_model
from sagsep.types import Instance, mask_key_for_ext_nodes, n_mask_words
from sagsep_bc_common.cvrpsep_binding import CvrpsepSeparator
from sagsep_bc_common.protocol import pin_cpu_threads

from .instance import CmstInstance

__all__ = [
    "SeparationRequest",
    "Separator",
    "CvrpsepCutSeparator",
    "SagSepTTSSeparator",
    "cut_mask_key",
    "preset_for_size",
]


@dataclass(frozen=True)
class SeparationRequest:
    support_u: np.ndarray
    support_v: np.ndarray
    support_x: np.ndarray
    iteration: int
    blocked: "frozenset[bytes]"


class Separator(Protocol):
    name: str

    def propose(self, request: SeparationRequest) -> Tuple[List[List[int]], List[float]]:
        pass

    def commit(self, accepted_indices: Sequence[int], retain: bool) -> None:
        pass

    def describe(self) -> str:
        pass

    def close(self) -> None:
        pass


def cut_mask_key(nodes_ext: Sequence[int], n_nodes: int) -> bytes:
    return mask_key_for_ext_nodes(nodes_ext, n_mask_words(int(n_nodes)), strip_depot=True)


def preset_for_size(n_customers: int) -> SagSepConfig:
    sizes = sorted(PRESET_SIZES, key=lambda tag: int(tag[1:]))
    for tag in sizes:
        if n_customers <= int(tag[1:]):
            return preset(tag)
    return preset(sizes[-1])


class CvrpsepCutSeparator:
    name = "cvrpsep"

    def __init__(
        self,
        instance: CmstInstance,
        *,
        max_cuts: int,
        min_violation: float,
        tree_commit: str = "offered",
        library_path: Optional[str | Path] = None,
    ) -> None:
        self._max_cuts = int(max_cuts)
        self._min_violation = float(min_violation)
        self._tree_commit = str(tree_commit)
        self._separator = CvrpsepSeparator(
            instance,
            max_cuts=max_cuts,
            min_violation=min_violation,
            library_path=library_path,
        )

    def propose(self, request: SeparationRequest) -> Tuple[List[List[int]], List[float]]:
        subsets, rhs_values, _max_violation = self._separator.separate(
            request.support_u, request.support_v, request.support_x
        )
        return subsets, rhs_values

    def commit(self, accepted_indices: Sequence[int], retain: bool) -> None:
        self._separator.commit(accepted_indices if retain else ())

    @property
    def pool_size(self) -> int:
        return self._separator.pool_size

    def describe(self) -> str:
        return (
            f"cvrpsep(max_cuts={self._max_cuts}, "
            f"min_violation={self._min_violation:g}, known_cuts={self.pool_size}, "
            f"tree_commit={self._tree_commit})"
        )

    def close(self) -> None:
        self._separator.close()


class SagSepTTSSeparator:
    name = "sagseptts"

    def __init__(
        self,
        instance: CmstInstance,
        *,
        run_dir: str | Path,
        device: str = "cpu",
        size_preset: Optional[str] = None,
        threads: int = 1,
        tree_commit: str = "offered",
    ) -> None:
        config = (
            preset(size_preset)
            if size_preset is not None
            else preset_for_size(instance.n_customers)
        )
        self.config = config
        self.device = device
        self.threads = max(1, int(threads))
        self.thread_detail = pin_cpu_threads(self.threads)
        self.tree_commit = str(tree_commit)
        n_nodes = int(instance.n_nodes)
        self._instance = Instance(
            name=instance.name,
            n_nodes=n_nodes,
            depot=0,
            coords=np.zeros((n_nodes, 2), dtype=np.float64),
            demand=instance.demand,
            capacity=int(instance.capacity),
            vehicles=1,
        )
        self._vehicle_replicas = features.vehicle_replica_count(self._instance)
        model = load_policy_model(run_dir, device=device)
        self._separator = LearnedSeparator(model, config, n_nodes=n_nodes)

    @property
    def vehicle_replicas(self) -> int:
        return self._vehicle_replicas

    def propose(self, request: SeparationRequest) -> Tuple[List[List[int]], List[float]]:
        cuts, _diagnostics, _tts_diag, _timings, _info = self._separator.propose(
            self._instance,
            iteration=request.iteration,
            blocked=request.blocked,
            support_u=request.support_u,
            support_v=request.support_v,
            support_x=request.support_x,
            vehicle_count=self._vehicle_replicas,
        )
        return [list(subset) for subset in cuts.sets], [float(r) for r in cuts.rhs_values]

    def commit(self, accepted_indices: Sequence[int], retain: bool) -> None:
        pass

    def describe(self) -> str:
        return (
            f"sagseptts(preset={self.config.size}, device={self.device}, "
            f"vehicle_replicas={self.vehicle_replicas}, threads={self.threads}, "
            f"tree_commit={self.tree_commit})"
        )

    def close(self) -> None:
        pass
