from __future__ import annotations

import contextlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sagsep.types import freeze_array

__all__ = [
    "GraphFeatures",
    "PolicyModel",
    "SubsetProbNet",
    "build_policy_network",
    "load_policy_model",
    "exportable_softmax",
]

EPS = 1e-8

# ---------------------------------------------------------------------------
# GPU-memory blocking (edge-dim E blocking primary, replica-dim K blocking as fallback)
# ---------------------------------------------------------------------------
_CHUNK_AUTO = -1
_CHUNK_OFF = 0
_CHUNK_DEFAULT_BYTES = 4 * 1024 * 1024 * 1024
_CHUNK_FREE_FRACTION = 0.35
_UNCHUNKED_EDGE_TENSORS = 4
_ECHUNK_EDGE_TENSORS = 2
_SOFTMAX_EDGE_TENSORS = 5
_NODE_TENSORS = 8

_CHECKPOINT_STEM = "subset_prob"

_matmul_precision_pinned = False


def _pin_matmul_precision() -> None:
    global _matmul_precision_pinned
    if _matmul_precision_pinned:
        return
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    _matmul_precision_pinned = True


def _read_chunk_knob(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "" or raw.strip() == "auto":
        return _CHUNK_AUTO
    try:
        value = int(raw.strip())
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(
            f"{name}={raw!r} is invalid ('auto' / 0=off / positive int = forced block size)"
        )
    return value


def _read_chunk_env() -> Tuple[int, int, int, bool]:
    echunk = _read_chunk_knob("SAGSEP_ECHUNK")
    kchunk = _read_chunk_knob("SAGSEP_KCHUNK")

    budget = _CHUNK_DEFAULT_BYTES
    raw_bytes = os.environ.get("SAGSEP_CHUNK_BYTES")
    if raw_bytes is not None and raw_bytes.strip() != "":
        try:
            value_b = int(raw_bytes.strip())
        except ValueError:
            value_b = -1
        if value_b < (1 << 20):
            raise ValueError(
                f"SAGSEP_CHUNK_BYTES={raw_bytes!r} is invalid (byte budget, at least 1 MiB)"
            )
        budget = value_b

    raw_dbg = os.environ.get("SAGSEP_KDEBUG")
    debug = bool(raw_dbg) and raw_dbg.strip() not in ("", "0")
    return echunk, kchunk, budget, debug


# ---------------------------------------------------------------------------
# GraphFeatures -- frozen model-input contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphFeatures:
    n_nodes: int
    vehicle_count: int
    edge_index: np.ndarray
    edge_attr: np.ndarray
    node_attr: np.ndarray

    def __post_init__(self) -> None:
        ei = np.asarray(self.edge_index, dtype=np.int64)
        ea = np.asarray(self.edge_attr, dtype=np.float32)
        na = np.asarray(self.node_attr, dtype=np.float32)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError("edge_index must be (2, E)")
        if ei.shape[1] == 0:
            raise ValueError("graph must contain at least one edge")
        if ea.ndim != 2 or ea.shape[0] != ei.shape[1]:
            raise ValueError("edge_attr must be (E, edge_dim)")
        if na.ndim != 3 or na.shape[0] != self.vehicle_count or na.shape[1] != self.n_nodes:
            raise ValueError("node_attr must be (K, n_nodes, node_dim)")
        object.__setattr__(self, "edge_index", freeze_array(ei))
        object.__setattr__(self, "edge_attr", freeze_array(ea))
        object.__setattr__(self, "node_attr", freeze_array(na))


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def _group_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    idx = dst.view((-1,) + (1,) * (scores.dim() - 1)).expand_as(scores)
    shape = (num_nodes,) + tuple(scores.shape[1:])
    src_max = scores.new_zeros(shape).scatter_reduce_(
        0, idx, scores.detach(), reduce="amax", include_self=False
    )
    out = scores - src_max.index_select(0, dst)
    out = out.exp()
    out_sum = out.new_zeros(shape).scatter_add_(0, idx, out) + 1e-16
    out_sum = out_sum.index_select(0, dst)
    return out / out_sum


def _group_softmax_exportable(
    scores: torch.Tensor, dst: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    idx = dst.view((-1,) + (1,) * (scores.dim() - 1)).expand_as(scores)
    shape = (num_nodes,) + tuple(scores.shape[1:])
    neg_inf = torch.full(shape, float("-inf"), dtype=scores.dtype, device=scores.device)
    src_max = neg_inf.scatter_reduce(0, idx, scores.detach(), reduce="amax", include_self=True)
    out = scores - src_max.index_select(0, dst)
    out = out.exp()
    out_sum = out.new_zeros(shape).scatter_add(0, idx, out) + 1e-16
    out_sum = out_sum.index_select(0, dst)
    return out / out_sum


@contextlib.contextmanager
def exportable_softmax():
    global _group_softmax
    original = _group_softmax
    _group_softmax = _group_softmax_exportable
    try:
        yield
    finally:
        _group_softmax = original


class _GraphTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(dim, num_heads, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self._scale = math.sqrt(self.head_dim + EPS)

    def forward_replicas(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_feat: torch.Tensor,
        edge_bias: torch.Tensor,
        edge_chunk: int = 0,
    ) -> torch.Tensor:
        n_nodes, n_rep, dim = node_feat.shape
        if self.training:
            edge_chunk = 0

        residual = node_feat
        h = self.norm1(node_feat)

        q = self.q_proj(h).view(n_nodes, n_rep, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(n_nodes, n_rep, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(n_nodes, n_rep, self.num_heads, self.head_dim)

        n_edges = int(dst.numel())
        if edge_chunk <= 0 or edge_chunk >= n_edges:
            q_dst = q.index_select(0, dst)
            k_src = k.index_select(0, src)
            v_src = v.index_select(0, src)

            scores = (q_dst * k_src).sum(dim=-1) / self._scale
            scores = scores + edge_bias.unsqueeze(1)

            weights = _group_softmax(scores, dst, n_nodes)
            weights = self.dropout(weights)
            weighted = (weights.unsqueeze(-1) * v_src).reshape(-1, n_rep, dim)
            idx = dst.view(-1, 1, 1).expand_as(weighted)
            attn_out = weighted.new_zeros((n_nodes, n_rep, dim)).scatter_add_(0, idx, weighted)
        else:
            attn_out = self._attention_edge_chunked(
                src, dst, q, k, v, edge_bias, edge_chunk, n_nodes, n_rep, dim, n_edges
            )
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        node_feat = residual + attn_out

        residual = node_feat
        node_feat = residual + self.ffn(self.norm2(node_feat))
        return node_feat

    def _attention_edge_chunked(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        edge_bias: torch.Tensor,
        edge_chunk: int,
        n_nodes: int,
        n_rep: int,
        dim: int,
        n_edges: int,
    ) -> torch.Tensor:
        scores = q.new_empty((n_edges, n_rep, self.num_heads))
        for start in range(0, n_edges, edge_chunk):
            width = min(edge_chunk, n_edges - start)
            q_dst = q.index_select(0, dst.narrow(0, start, width))
            k_src = k.index_select(0, src.narrow(0, start, width))
            block = (q_dst * k_src).sum(dim=-1) / self._scale
            block = block + edge_bias.narrow(0, start, width).unsqueeze(1)
            scores.narrow(0, start, width).copy_(block)
            del q_dst, k_src, block

        weights = _group_softmax(scores, dst, n_nodes)
        del scores

        attn_out = q.new_zeros((n_nodes, n_rep, dim))
        for start in range(0, n_edges, edge_chunk):
            width = min(edge_chunk, n_edges - start)
            dst_blk = dst.narrow(0, start, width)
            v_src = v.index_select(0, src.narrow(0, start, width))
            weighted = (weights.narrow(0, start, width).unsqueeze(-1) * v_src).reshape(
                -1, n_rep, dim
            )
            attn_out.scatter_add_(0, dst_blk.view(-1, 1, 1).expand_as(weighted), weighted)
            del v_src, weighted
        return attn_out


class _GraphTransformer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.node_encoder = nn.Linear(node_dim, latent_dim)
        self.edge_encoder = nn.Linear(edge_dim, latent_dim)
        self.layers = nn.ModuleList(
            [
                _GraphTransformerLayer(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.edge_update = nn.Linear(2 * latent_dim, latent_dim)

    def forward_replicas(
        self,
        edge_index: torch.Tensor,
        nf_stacked: torch.Tensor,
        ef: torch.Tensor,
        edge_chunk: int = 0,
    ) -> torch.Tensor:
        target_dtype = self.node_encoder.weight.dtype
        node_feat = self.node_encoder(nf_stacked.to(target_dtype))
        src, dst = edge_index[0], edge_index[1]

        edge_feat = self.edge_encoder(ef.to(target_dtype))

        for layer in self.layers:
            node_feat = layer.forward_replicas(
                src, dst, node_feat, layer.edge_proj(edge_feat), edge_chunk
            )
        return node_feat


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: Optional[Tuple[int, ...]] = None,
        hidden_act: str = "ReLU",
        out_act: str = "Identity",
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden) if hidden is not None else [64, 32]
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()
        layer_dims = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(i, o, bias=bias) for i, o in zip(layer_dims[:-1], layer_dims[1:])]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.layers[:-1]:
            x = self.hidden_act(linear(x))
        return self.out_act(self.layers[-1](x))


class _BackboneShell(nn.Module):
    def __init__(self, inner: _GraphTransformer) -> None:
        super().__init__()
        self.backbone = inner


class _SubsetProbNet(nn.Module):
    def __init__(self, gt: _GraphTransformer, policy: _MLP) -> None:
        super().__init__()
        self.backbone = _BackboneShell(gt)
        self.policy = policy

    def forward_replicas(
        self,
        edge_index: torch.Tensor,
        nf_stacked: torch.Tensor,
        ef: torch.Tensor,
        edge_chunk: int = 0,
    ) -> torch.Tensor:
        node_latent = self.backbone.backbone.forward_replicas(
            edge_index, nf_stacked, ef, edge_chunk
        )
        n_nodes, n_rep, dim = node_latent.shape
        logits = self.policy(node_latent.reshape(n_nodes * n_rep, dim))
        return logits.reshape(n_nodes, n_rep, 1)

    def forward_batch(
        self,
        edge_index: torch.Tensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward_replicas(edge_index, node_attr[:, None, :], edge_attr, 0)
        return logits.reshape(int(node_attr.shape[0]), 1)


def _build_net(model_cfg: Dict[str, object]) -> _SubsetProbNet:
    allowed = {
        "node_dim",
        "edge_dim",
        "latent_dim",
        "n_layer",
        "backbone_type",
        "transformer_params",
        "policy_params",
        "debug",
    }
    unknown = set(model_cfg) - allowed
    if unknown:
        raise ValueError(f"Unsupported model config keys: {sorted(unknown)}")

    backbone_type = str(model_cfg.get("backbone_type", "transformer"))
    if backbone_type != "transformer":
        raise NotImplementedError(
            f"SAG-Sep distils the transformer backbone only, got {backbone_type!r}"
        )

    tp = dict(model_cfg.get("transformer_params") or {})
    m_band_mode = str(tp.pop("m_band_mode", "concat"))
    if m_band_mode != "concat":
        raise NotImplementedError(
            f"only m_band_mode='concat' is distilled (the frozen recipe), got {m_band_mode!r}"
        )
    tp_allowed = {"num_heads", "dropout", "use_edge_features"}
    tp_unknown = set(tp) - tp_allowed
    if tp_unknown:
        raise ValueError(f"Unsupported transformer_params keys: {sorted(tp_unknown)}")
    if not bool(tp.get("use_edge_features", True)):
        raise NotImplementedError(
            "only use_edge_features=True (the released checkpoint) is distilled; the "
            "edge-free attention path is not part of the frozen recipe"
        )

    latent_dim = int(model_cfg["latent_dim"])
    gt = _GraphTransformer(
        node_dim=int(model_cfg["node_dim"]),
        edge_dim=int(model_cfg["edge_dim"]),
        latent_dim=latent_dim,
        num_layers=int(model_cfg["n_layer"]),
        num_heads=int(tp.get("num_heads", 8)),
        dropout=float(tp.get("dropout", 0.1)),
    )

    pp = dict(model_cfg["policy_params"])
    pp_allowed = {"hidden", "hidden_act", "out_act", "bias"}
    pp_unknown = set(pp) - pp_allowed
    if pp_unknown:
        raise ValueError(f"Unsupported policy_params keys: {sorted(pp_unknown)}")
    policy = _MLP(
        input_dim=latent_dim,
        output_dim=1,
        hidden=tuple(pp["hidden"]) if pp.get("hidden") is not None else None,
        hidden_act=str(pp.get("hidden_act", "ReLU")),
        out_act=str(pp.get("out_act", "Identity")),
        bias=bool(pp.get("bias", True)),
    )
    return _SubsetProbNet(gt, policy)

SubsetProbNet = _SubsetProbNet


def build_policy_network(model_cfg: Mapping[str, object]) -> SubsetProbNet:
    return _build_net(dict(model_cfg))


# ---------------------------------------------------------------------------
# PolicyModel -- public inference wrapper
# ---------------------------------------------------------------------------


class PolicyModel:
    def __init__(self, net: _SubsetProbNet, device: torch.device, checkpoint_path: str) -> None:
        self._net = net
        self._torch_device = device
        self.device: str = str(device)
        self.checkpoint_path = checkpoint_path
        self._pin = device.type == "cuda"
        self._cpu_buffers: Dict[str, torch.Tensor] = {}
        self._dev_buffers: Dict[str, torch.Tensor] = {}
        (
            self._echunk_cfg,
            self._kchunk_cfg,
            self._chunk_bytes,
            self._chunk_debug,
        ) = _read_chunk_env()
        self._echunk_cap = 0
        self._kchunk_cap = 0
        self._chunk_logged: Tuple[int, int] = (0, 0)
        self._latent_dim = int(net.backbone.backbone.latent_dim)
        self._num_heads = int(net.backbone.backbone.layers[0].num_heads)

    @classmethod
    def from_network(cls, net: SubsetProbNet, device: str = "cpu") -> "PolicyModel":
        _pin_matmul_precision()
        torch_device = _resolve_device(device)
        net.to(torch_device)
        return cls(net=net, device=torch_device, checkpoint_path="<in-memory>")

    def _budget_bytes(self) -> int:
        budget = self._chunk_bytes
        try:
            free, _total = torch.cuda.mem_get_info(self._torch_device)
            cached_free = torch.cuda.memory_reserved(
                self._torch_device
            ) - torch.cuda.memory_allocated(self._torch_device)
            usable = int(free) + max(0, int(cached_free))
        except Exception:
            usable = 0
        dynamic = int(usable * _CHUNK_FREE_FRACTION)
        if dynamic > 0:
            budget = min(budget, dynamic)
        return budget

    def _fixed_bytes(self, n_nodes: int, n_rep: int, n_edges: int) -> int:
        d = self._latent_dim
        h = self._num_heads
        return (
            _SOFTMAX_EDGE_TENSORS * n_edges * n_rep * h * 4
            + _NODE_TENSORS * n_nodes * n_rep * d * 4
        )

    def _decide_kchunk(self, n_nodes: int, n_rep: int, n_edges: int) -> int:
        if self._kchunk_cfg == _CHUNK_OFF:
            return n_rep
        if self._kchunk_cfg > 0:
            block = self._kchunk_cfg
        else:
            if self._torch_device.type != "cuda":
                return n_rep
            per_replica = self._fixed_bytes(n_nodes, 1, n_edges)
            if per_replica <= 0:
                return n_rep
            budget = self._budget_bytes()
            if per_replica * n_rep <= budget:
                return n_rep
            cap = max(1, budget // per_replica)
            n_blocks = (n_rep + cap - 1) // cap
            block = (n_rep + n_blocks - 1) // n_blocks
        if self._kchunk_cap > 0:
            block = min(block, self._kchunk_cap)
        return max(1, min(int(block), n_rep))

    def _decide_echunk(self, n_nodes: int, n_rep: int, n_edges: int) -> int:
        if self._echunk_cfg == _CHUNK_OFF:
            return n_edges
        if self._echunk_cfg > 0:
            block = self._echunk_cfg
        else:
            if self._torch_device.type != "cuda":
                return n_edges
            budget = self._budget_bytes()
            fixed = self._fixed_bytes(n_nodes, n_rep, n_edges)
            per_edge_unchunked = _UNCHUNKED_EDGE_TENSORS * n_rep * self._latent_dim * 4
            if fixed + per_edge_unchunked * n_edges <= budget:
                return n_edges
            per_edge = _ECHUNK_EDGE_TENSORS * n_rep * self._latent_dim * 4
            avail = budget - fixed
            if avail < per_edge:
                block = 1
            else:
                cap = avail // per_edge
                n_blocks = max(2, (n_edges + cap - 1) // cap)
                block = (n_edges + n_blocks - 1) // n_blocks
        if self._echunk_cap > 0:
            block = min(block, self._echunk_cap)
        return max(1, min(int(block), n_edges))

    def _log_chunk(self, n_nodes: int, n_rep: int, n_edges: int, ec: int, kb: int) -> None:
        active = ec < n_edges or kb < n_rep
        if not (self._chunk_debug or (active and (ec, kb) != self._chunk_logged)):
            return
        peak = (
            _ECHUNK_EDGE_TENSORS * ec * kb * self._latent_dim * 4
            + self._fixed_bytes(n_nodes, kb, n_edges)
        ) / float(1 << 30)
        print(
            f"[sagsep/model] chunk: N={n_nodes} K={n_rep} E={n_edges} -> "
            f"Ec={ec} ({(n_edges + ec - 1) // ec} eblk) Kb={kb} "
            f"({(n_rep + kb - 1) // kb} kblk), ~{peak:.2f} GiB/layer",
            file=sys.stderr,
        )
        self._chunk_logged = (ec, kb)

    def _forward_replicas_chunked(
        self,
        edge_index: torch.Tensor,
        nf_stacked: torch.Tensor,
        ef: torch.Tensor,
        n_edges: int,
    ) -> torch.Tensor:
        n_nodes, n_rep, _ = nf_stacked.shape
        kblock = self._decide_kchunk(n_nodes, n_rep, n_edges)
        eblock = self._decide_echunk(n_nodes, min(kblock, n_rep), n_edges)
        self._log_chunk(n_nodes, n_rep, n_edges, eblock, kblock)

        if kblock >= n_rep:
            return self._run_block(edge_index, nf_stacked, ef, n_edges, eblock)

        parts = []
        k0 = 0
        while k0 < n_rep:
            width = min(kblock, n_rep - k0)
            parts.append(
                self._run_block(
                    edge_index,
                    nf_stacked.narrow(1, k0, width).contiguous(),
                    ef,
                    n_edges,
                    self._decide_echunk(n_nodes, width, n_edges),
                )
            )
            k0 += width
        return torch.cat(parts, dim=1)

    def _run_block(
        self,
        edge_index: torch.Tensor,
        nf_block: torch.Tensor,
        ef: torch.Tensor,
        n_edges: int,
        eblock: int,
    ) -> torch.Tensor:
        while True:
            try:
                return self._net.forward_replicas(
                    edge_index, nf_block, ef, 0 if eblock >= n_edges else eblock
                )
            except torch.OutOfMemoryError:
                if self._echunk_cfg == _CHUNK_OFF or eblock <= 1:
                    raise
                nxt = max(1, min(eblock, n_edges) // 2)
                print(
                    f"[sagsep/model] chunk: OOM at E={n_edges} Ec={eblock}; "
                    f"retrying at Ec={nxt}",
                    file=sys.stderr,
                )
                eblock = nxt
                self._echunk_cap = nxt
                self._chunk_logged = (0, 0)
                torch.cuda.empty_cache()

    def _staged(self, name: str, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        shape = tuple(array.shape)
        buf = self._cpu_buffers.get(name)
        if buf is None or tuple(buf.shape) != shape or buf.dtype != dtype:
            buf = torch.empty(shape, dtype=dtype, pin_memory=self._pin)
            self._cpu_buffers[name] = buf
        buf.numpy()[...] = array
        if self._torch_device.type == "cpu":
            return buf
        dev = self._dev_buffers.get(name)
        if dev is None or tuple(dev.shape) != shape or dev.dtype != dtype:
            dev = torch.empty(shape, dtype=dtype, device=self._torch_device)
            self._dev_buffers[name] = dev
        dev.copy_(buf, non_blocking=True)
        return dev

    def forward_probs(self, feats: GraphFeatures) -> np.ndarray:
        n_rep = int(feats.vehicle_count)
        n_nodes = int(feats.n_nodes)
        with torch.no_grad():
            edge_index = self._staged("edge_index", feats.edge_index, torch.int64)
            ef = self._staged("edge_attr", feats.edge_attr, torch.float32)
            nf_stacked = self._staged(
                "node_attr", feats.node_attr.transpose(1, 0, 2), torch.float32
            )
            logits = self._forward_replicas_chunked(
                edge_index, nf_stacked, ef, int(feats.edge_index.shape[1])
            )
            logits = logits.permute(1, 0, 2).reshape(-1).detach().cpu()
            probs = torch.sigmoid(logits)
        return probs.view(n_rep, n_nodes).numpy()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _resolve_checkpoint_file(run_path: Path) -> Path:
    target = run_path / f"{_CHECKPOINT_STEM}_best.pt"
    if not target.exists():
        raise FileNotFoundError(f"No {target.name} in {run_path}")
    return target


def _load_state_dict(checkpoint: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def _resolve_device(requested: str) -> torch.device:
    requested = (requested or "cpu").lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[set_device] CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        index = None
        if ":" in requested:
            suffix = requested.split(":", 1)[1]
            if suffix:
                try:
                    index = int(suffix)
                except ValueError:
                    pass
        device = torch.device("cuda", index) if index is not None else torch.device("cuda")
        try:
            probe = torch.zeros((2, 2), device=device)
            _ = torch.nn.functional.layer_norm(probe, (2,))
            torch.cuda.synchronize(device)
            return device
        except Exception as exc:
            print(
                "[set_device] CUDA device "
                f"{device} is visible but unusable for policy inference ({exc}); "
                "falling back to CPU."
            )
            return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    try:
        return torch.device(requested)
    except (RuntimeError, ValueError):
        print(f"[set_device] Unknown device '{requested}'. Falling back to CPU.")
        return torch.device("cpu")

_GRAPH_BUILDER_ID = "partial_2hop_max_min_v1"


def _validate_input_contract(
    contract: object,
    config_path: Path,
    partial_connect_cross_depot: Optional[bool],
) -> None:
    if not isinstance(contract, dict):
        raise ValueError(
            f"input_contract in {config_path} must be a mapping, got "
            f"{type(contract).__name__}"
        )
    builder = contract.get("graph_builder")
    if builder != _GRAPH_BUILDER_ID:
        raise ValueError(
            f"input_contract.graph_builder in {config_path} must be "
            f"{_GRAPH_BUILDER_ID!r}, got {builder!r}: these weights were trained "
            "against a different graph construction than sagsep.features builds"
        )
    if contract.get("m_value_closed_interval") is not True:
        raise ValueError(
            f"input_contract.m_value_closed_interval in {config_path} must be true, "
            f"got {contract.get('m_value_closed_interval')!r}: the vehicle-replica "
            "sweep of this package is the closed interval, m = 1..K"
        )
    if partial_connect_cross_depot is None:
        return
    want = bool(partial_connect_cross_depot)
    got = contract.get("partial_connect_cross_depot")
    if not isinstance(got, bool) or got != want:
        raise ValueError(
            f"input_contract.partial_connect_cross_depot in {config_path} is {got!r}, "
            f"but the runtime feature config asks for {want!r}: a checkpoint has to be "
            "served with the cross-depot setting it was trained with"
        )


def load_policy_model(
    run_dir: str | Path,
    *,
    device: str = "cuda:0",
    partial_connect_cross_depot: Optional[bool] = None,
) -> PolicyModel:
    _pin_matmul_precision()

    run_path = Path(run_dir).resolve()
    if run_path.is_file():
        checkpoint = run_path
        config_path = run_path.with_suffix(".yaml")
    else:
        checkpoint = _resolve_checkpoint_file(run_path)
        config_path = run_path / "config_snapshot.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config snapshot not found: {config_path}")
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        snapshot = yaml.safe_load(fh)

    if snapshot.get("input_contract") is not None:
        _validate_input_contract(
            snapshot["input_contract"], config_path, partial_connect_cross_depot
        )

    net = _build_net(dict(snapshot["model"]))
    net.load_state_dict(_load_state_dict(checkpoint))

    trainer_cfg = dict(snapshot.get("trainer") or {})
    train_precision = str(trainer_cfg.get("precision", "fp32")).lower()
    if train_precision != "fp32":
        raise ValueError(
            "trainer.precision must be 'fp32' in SAG-Sep, got "
            f"{trainer_cfg.get('precision')!r} in {config_path} "
            "(the fp64 training arm is out of the frozen recipe)"
        )

    torch_device = _resolve_device(device)
    net.to(torch_device)
    net.eval()
    net.requires_grad_(False)

    return PolicyModel(net=net, device=torch_device, checkpoint_path=str(checkpoint))
