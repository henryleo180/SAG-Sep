from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = ["op_histogram", "rewrite"]

_ND_SHAPE_INIT = "sagsep_nd_rowidx_shape"

DEFAULT_REDUCTIONS: Tuple[str, ...] = ("add", "max")


def op_histogram(model: Any) -> "collections.Counter[str]":
    import onnx

    counts: "collections.Counter[str]" = collections.Counter()

    def walk(graph: Any, prefix: str = "") -> None:
        for node in graph.node:
            counts[prefix + node.op_type] += 1
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    walk(attr.g, "<sub>")
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for sub in attr.graphs:
                        walk(sub, "<sub>")

    walk(model.graph)
    return counts


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------


def _producers(graph: Any) -> Dict[str, Any]:
    return {out: node for node in graph.node for out in node.output}


def _attr(node: Any, name: str) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            return attr
    return None


def _attr_i(node: Any, name: str, default: int) -> int:
    attr = _attr(node, name)
    return default if attr is None else int(attr.i)


def _attr_s(node: Any, name: str, default: str) -> str:
    attr = _attr(node, name)
    return default if attr is None else attr.s.decode()


def _const_array(graph: Any, prod: Dict[str, Any], name: str) -> Optional[np.ndarray]:
    from onnx import numpy_helper

    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    node = prod.get(name)
    if node is not None and node.op_type == "Constant":
        attr = _attr(node, "value")
        if attr is not None:
            return numpy_helper.to_array(attr.t)
    return None


def _rank_map(model: Any) -> Dict[str, int]:
    import onnx

    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        return {}
    ranks: Dict[str, int] = {}
    graph = inferred.graph
    for info in list(graph.value_info) + list(graph.input) + list(graph.output):
        tensor = info.type.tensor_type
        if tensor.HasField("shape"):
            ranks[info.name] = len(tensor.shape.dim)
    return ranks


def _rewire(graph: Any, mapping: Dict[str, str]) -> None:
    import onnx

    def walk(sub: Any) -> None:
        for node in sub.node:
            for i, name in enumerate(node.input):
                if name in mapping:
                    node.input[i] = mapping[name]
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    walk(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for inner in attr.graphs:
                        walk(inner)
        for out in sub.output:
            if out.name in mapping:
                out.name = mapping[out.name]

    walk(graph)


def _subgraph_refs(node: Any) -> List[str]:
    import onnx

    refs: List[str] = []

    def walk(sub: Any) -> None:
        local = {o for n in sub.node for o in n.output} | {i.name for i in sub.input}
        for inner in sub.node:
            for name in inner.input:
                if name and name not in local:
                    refs.append(name)
            for attr in inner.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    walk(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for deeper in attr.graphs:
                        walk(deeper)

    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            walk(attr.g)
        elif attr.type == onnx.AttributeProto.GRAPHS:
            for sub in attr.graphs:
                walk(sub)
    return refs


def _dce(graph: Any) -> int:
    prod = _producers(graph)
    live: Set[str] = set()
    stack: List[str] = [out.name for out in graph.output]
    while stack:
        name = stack.pop()
        if not name or name in live:
            continue
        live.add(name)
        node = prod.get(name)
        if node is None:
            continue
        for dep in list(node.input) + _subgraph_refs(node):
            if dep not in live:
                stack.append(dep)

    keep = [node for node in graph.node if any(o in live for o in node.output)]
    removed = len(graph.node) - len(keep)
    del graph.node[:]
    graph.node.extend(keep)

    used = {i for node in graph.node for i in node.input}
    kept_init = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_init)
    return removed


# ---------------------------------------------------------------------------
# pass A: eliminate the statically-false rank-0 guard If
# ---------------------------------------------------------------------------


def _eliminate_degenerate_if(graph: Any, ranks: Dict[str, int]) -> int:
    prod = _producers(graph)
    removed = 0
    survivors: List[Any] = []
    mapping: Dict[str, str] = {}

    for node in graph.node:
        if node.op_type != "If":
            survivors.append(node)
            continue

        equal = prod.get(node.input[0])
        ok = equal is not None and equal.op_type == "Equal"
        size = prod.get(equal.input[0]) if ok else None
        ok = ok and size is not None and size.op_type == "Size"
        shape = prod.get(size.input[0]) if ok else None
        ok = ok and shape is not None and shape.op_type == "Shape"
        if ok:
            rhs = _const_array(graph, prod, equal.input[1])
            ok = rhs is not None and rhs.size == 1 and int(rhs.reshape(-1)[0]) == 0
        if ok:
            rank = ranks.get(shape.input[0])
            ok = rank is not None and rank >= 1

        then_attr = _attr(node, "then_branch")
        else_attr = _attr(node, "else_branch")
        ok = ok and then_attr is not None and else_attr is not None
        srcs: List[str] = []
        if ok:
            else_graph = else_attr.g
            ok = (
                all(n.op_type == "Identity" for n in else_graph.node)
                and len(else_graph.output) == len(node.output)
                and all(
                    n.op_type in ("Constant", "Reshape", "Squeeze")
                    for n in then_attr.g.node
                )
            )
            if ok:
                src_of = {n.output[0]: n.input[0] for n in else_graph.node}
                try:
                    srcs = [src_of[o.name] for o in else_graph.output]
                except KeyError:
                    ok = False
        if not ok:
            survivors.append(node)
            continue

        for out_name, src in zip(node.output, srcs):
            mapping[out_name] = src
        removed += 1

    if removed:
        del graph.node[:]
        graph.node.extend(survivors)
        for _ in range(len(mapping) + 1):
            changed = False
            for key, value in list(mapping.items()):
                if value in mapping and mapping[value] != value:
                    mapping[key] = mapping[value]
                    changed = True
            if not changed:
                break
        _rewire(graph, mapping)
    return removed


# ---------------------------------------------------------------------------
# pass B: ScatterElements(axis=0, indices=Expand(row_idx)) -> ScatterND
# ---------------------------------------------------------------------------


def _row_index_base(graph: Any, prod: Dict[str, Any], expand: Any, rank: int) -> Optional[str]:
    reshape = prod.get(expand.input[0])
    if reshape is None or reshape.op_type != "Reshape":
        return None
    shape = _const_array(graph, prod, reshape.input[1])
    if shape is None:
        return None
    dims = [int(v) for v in np.asarray(shape).reshape(-1)]
    if len(dims) != rank or dims[0] != -1 or any(v != 1 for v in dims[1:]):
        return None
    return reshape.input[0]


def _unwrap_identity_slice(
    graph: Any, prod: Dict[str, Any], updates: str, expand_out: str
) -> str:
    node = prod.get(updates)
    if node is None or node.op_type != "Slice" or len(node.input) != 3:
        return updates
    starts = _const_array(graph, prod, node.input[1])
    if starts is None or np.any(np.asarray(starts).reshape(-1) != 0):
        return updates
    ends = prod.get(node.input[2])
    if ends is None or ends.op_type != "Shape" or ends.input[0] != expand_out:
        return updates
    if _attr(ends, "start") is not None or _attr(ends, "end") is not None:
        return updates
    return node.input[0]


def _scatter_to_scatternd(graph: Any, reductions: Sequence[str]) -> Tuple[int, List[str]]:
    from onnx import helper, numpy_helper

    prod = _producers(graph)
    idx2d: Dict[str, str] = {}
    new_idx_nodes: Dict[str, Any] = {}
    converted: List[str] = []
    out_nodes: List[Any] = []

    for node in graph.node:
        if node.op_type != "ScatterElements" or _attr_i(node, "axis", 0) != 0:
            out_nodes.append(node)
            continue
        reduction = _attr_s(node, "reduction", "none")
        if reduction not in reductions:
            out_nodes.append(node)
            continue
        expand = prod.get(node.input[1])
        if expand is None or expand.op_type != "Expand":
            out_nodes.append(node)
            continue
        reshape = prod.get(expand.input[0])
        if reshape is None or reshape.op_type != "Reshape":
            out_nodes.append(node)
            continue
        shape = _const_array(graph, prod, reshape.input[1])
        if shape is None:
            out_nodes.append(node)
            continue
        base = _row_index_base(graph, prod, expand, int(np.asarray(shape).size))
        if base is None:
            out_nodes.append(node)
            continue

        if base not in idx2d:
            name = f"sagsep_nd_rowidx_{len(idx2d)}"
            idx2d[base] = name
            new_idx_nodes[base] = helper.make_node(
                "Reshape", [base, _ND_SHAPE_INIT], [name], name=f"{name}_reshape"
            )
        updates = _unwrap_identity_slice(graph, prod, node.input[2], expand.output[0])
        out_nodes.append(
            helper.make_node(
                "ScatterND",
                [node.input[0], idx2d[base], updates],
                list(node.output),
                name=(node.name or "ScatterElements") + "_nd",
                reduction=reduction,
            )
        )
        converted.append(node.name)

    if not converted:
        return 0, []

    if not any(init.name == _ND_SHAPE_INIT for init in graph.initializer):
        graph.initializer.append(
            numpy_helper.from_array(np.asarray([-1, 1], dtype=np.int64), _ND_SHAPE_INIT)
        )

    final: List[Any] = []
    pending = dict(new_idx_nodes)
    produced: Set[str] = {i.name for i in graph.input} | {i.name for i in graph.initializer}
    for node in out_nodes:
        final.append(node)
        produced.update(node.output)
        for base in [b for b in pending if b in produced]:
            final.append(pending.pop(base))
    if pending:
        final = list(pending.values()) + final
    del graph.node[:]
    graph.node.extend(final)
    return len(converted), converted


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def rewrite(
    model: Any,
    *,
    reductions: Sequence[str] = DEFAULT_REDUCTIONS,
    verbose: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    import onnx

    before = op_histogram(model)
    graph = model.graph

    ranks = _rank_map(model)
    n_if = _eliminate_degenerate_if(graph, ranks)
    n_nd, names = _scatter_to_scatternd(graph, reductions)
    n_dce = _dce(graph)

    alive = {out for node in graph.node for out in node.output}
    kept = [info for info in graph.value_info if info.name in alive]
    del graph.value_info[:]
    graph.value_info.extend(kept)

    after = op_histogram(model)
    gained = after.get("ScatterND", 0) - before.get("ScatterND", 0)
    if gained != n_nd:
        raise RuntimeError(
            f"expected +{n_nd} ScatterND, graph gained {gained} "
            f"({before.get('ScatterND', 0)} -> {after.get('ScatterND', 0)})"
        )
    onnx.checker.check_model(model)

    stats: Dict[str, Any] = {
        "if_eliminated": n_if,
        "scatternd_converted": n_nd,
        "converted_names": names,
        "dce_removed": n_dce,
        "before": dict(before),
        "after": dict(after),
    }
    if verbose:
        print(f"[scatter] If eliminated       : {n_if}")
        print(f"[scatter] ScatterElements->ND : {n_nd}")
        print(f"[scatter] dead nodes removed  : {n_dce}")
        for key in sorted(set(before) | set(after)):
            b, a = before.get(key, 0), after.get(key, 0)
            if b != a:
                print(f"[scatter] {key:<22}{b:>8}{a:>8}   {a - b:+d}")
    return model, stats
