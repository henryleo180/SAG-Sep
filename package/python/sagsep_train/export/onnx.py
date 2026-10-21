from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sagsep.model import GraphFeatures, build_policy_network, exportable_softmax
from sagsep_train.export import parity as parity_mod
from sagsep_train.export import scatter as scatter_mod

__all__ = [
    "INPUT_NAMES",
    "OUTPUT_NAME",
    "REFERENCE_OPSET",
    "export_onnx",
    "graph_meta",
    "load_export_network",
]

INPUT_NAMES: Tuple[str, str, str] = parity_mod.INPUT_NAMES
OUTPUT_NAME: str = parity_mod.OUTPUT_NAME

REFERENCE_OPSET = 18



# ---------------------------------------------------------------------------
# exportable group softmax
# ---------------------------------------------------------------------------


class _ExportableSoftmax:
    def __enter__(self) -> "_ExportableSoftmax":
        self._cm = exportable_softmax()
        self._cm.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._cm.__exit__(*exc)


# ---------------------------------------------------------------------------
# export wrapper
# ---------------------------------------------------------------------------


class _ExportWrapper(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(
        self,
        edge_index: torch.Tensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        nf_stacked = node_attr.permute(1, 0, 2)
        logits = self.net.forward_replicas(edge_index, nf_stacked, edge_attr, 0)
        logits = logits.permute(1, 0, 2).reshape(-1)
        probs = torch.sigmoid(logits)
        return probs.view(node_attr.shape[0], node_attr.shape[1])


def load_export_network(ckpt_path: str | Path) -> nn.Module:
    import yaml

    path = Path(ckpt_path)
    manifest_path = path.with_suffix(".yaml")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"model manifest not found next to the checkpoint: {manifest_path.name}"
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    net = build_policy_network(dict(manifest["model"]))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    net.load_state_dict(state_dict)
    net.eval()
    net.requires_grad_(False)
    return net


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _as_network(net_or_ckpt: Any) -> nn.Module:
    if isinstance(net_or_ckpt, nn.Module):
        return net_or_ckpt
    return load_export_network(net_or_ckpt)


def _sample_args(feats: GraphFeatures) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(feats.edge_index, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(feats.node_attr, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(feats.edge_attr, dtype=np.float32)),
    )


def _assert_bitwise(name: str, what: str, ref: np.ndarray, got: np.ndarray) -> None:
    if ref.tobytes() == got.tobytes():
        return
    worst = int(np.argmax(np.abs(ref.astype(np.float64) - got.astype(np.float64))))
    raise RuntimeError(
        f"{what} is NOT bitwise identical to the frozen forward on case {name} "
        f"(worst element at flat index {worst}: {ref.ravel()[worst]!r} vs {got.ravel()[worst]!r})"
    )


def graph_meta(path: str | Path) -> Dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    opsets = {entry.domain or "": int(entry.version) for entry in model.opset_import}
    return {
        "opset": opsets.get("", None),
        "opset_imports": opsets,
        "ir_version": int(model.ir_version),
        "n_nodes": len(model.graph.node),
        "op_histogram": dict(sorted(scatter_mod.op_histogram(model).items())),
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
    }


def export_onnx(
    net_or_ckpt: Any,
    out_path: str | Path,
    *,
    opset: Optional[int] = None,
    scatternd: bool = True,
    cases: Optional[Mapping[str, GraphFeatures]] = None,
    keep_raw: bool = False,
    verbose: bool = False,
    stats: Optional[MutableMapping[str, Any]] = None,
) -> Path:
    import onnx

    net = _as_network(net_or_ckpt)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    opset_version = REFERENCE_OPSET if opset is None else int(opset)
    case_map = dict(parity_mod.builtin_cases() if cases is None else cases)
    if not case_map:
        raise ValueError("export needs at least one case for the bitwise gates")
    record: MutableMapping[str, Any] = {} if stats is None else stats
    record["opset"] = opset_version
    record["cases"] = sorted(case_map)

    wrapper = _ExportWrapper(net).eval()

    for name, feats in case_map.items():
        reference = parity_mod.torch_probs(net, feats)
        with _ExportableSoftmax():
            patched = parity_mod.torch_probs(net, feats)
            with torch.no_grad():
                traced = wrapper(*_sample_args(feats)).numpy()
        _assert_bitwise(name, "the exportable group-softmax rewrite", reference, patched)
        _assert_bitwise(name, "the export wrapper", reference, traced)
    if verbose:
        print(f"[export] frozen-forward parity on {len(case_map)} cases: bitwise OK")

    sample = case_map[next(iter(case_map))]
    dynamic_axes = {
        "edge_index": {1: "E"},
        "node_attr": {0: "K", 1: "N"},
        "edge_attr": {0: "E"},
        OUTPUT_NAME: {0: "K", 1: "N"},
    }
    raw_path = out.with_name(out.stem + "_scatterelements_raw.onnx")
    with torch.no_grad(), _ExportableSoftmax(), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            _sample_args(sample),
            str(raw_path),
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

    model = onnx.load(str(raw_path))
    onnx.checker.check_model(model)
    record["n_nodes_raw"] = len(model.graph.node)
    record["ir_version"] = int(model.ir_version)

    if scatternd:
        model, rewrite_stats = scatter_mod.rewrite(model, verbose=verbose)
        record["scatternd"] = {k: v for k, v in rewrite_stats.items() if k != "converted_names"}
        histogram = scatter_mod.op_histogram(model)
        leftover = {
            op: histogram.get(op, 0)
            for op in ("ScatterElements", "Expand")
            if histogram.get(op, 0)
        }
        if leftover:
            raise RuntimeError(
                f"the ScatterND rewrite left {leftover} in the graph: the memory blow-up "
                "it exists to remove would still be there"
            )
        onnx.save(model, str(out))
        raw_session = new_session = None
        try:
            raw_session = parity_mod.ort_session(raw_path)
            new_session = parity_mod.ort_session(out)
            for name, feats in case_map.items():
                before = parity_mod.ort_probs(raw_session, feats)
                after = parity_mod.ort_probs(new_session, feats)
                if before.tobytes() != after.tobytes():
                    diff = float(
                        np.abs(before.astype(np.float64) - after.astype(np.float64)).max()
                    )
                    raise RuntimeError(
                        "the ScatterND rewrite is NOT bitwise identical to the traced "
                        f"graph on case {name} (max|delta| = {diff:.3e})"
                    )
        finally:
            del raw_session, new_session
        if verbose:
            print(f"[export] ScatterND rewrite on {len(case_map)} cases: bitwise OK")
    else:
        onnx.save(model, str(out))

    if keep_raw:
        record["raw_path"] = str(raw_path.name)
    else:
        raw_path.unlink()
    record["n_nodes"] = len(model.graph.node)
    record["bytes"] = out.stat().st_size
    record["op_histogram"] = dict(sorted(scatter_mod.op_histogram(model).items()))
    if verbose:
        print(f"[export] wrote {out.name} ({record['bytes']} bytes, {record['n_nodes']} nodes)")
    return out
