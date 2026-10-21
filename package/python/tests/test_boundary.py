from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
SERVED_DIR = PKG_ROOT / "sagsep"
TRAIN_DIR = PKG_ROOT / "sagsep_train"

BANNED_TRAIN_IMPORTS = ("torch_geometric", "torch_scatter")


def _python_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_roots(tree: ast.Module) -> Iterator[Tuple[ast.AST, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            yield node, node.module or ""


def _top_level(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _rel(path: Path) -> str:
    return str(path.relative_to(PKG_ROOT))


def test_served_package_does_not_import_the_training_package() -> None:
    files = _python_files(SERVED_DIR)
    assert files, f"no python sources found under {SERVED_DIR}"
    offenders: List[str] = []
    for path in files:
        for node, module in _imported_roots(_parse(path)):
            if _top_level(module) == "sagsep_train":
                offenders.append(f"{_rel(path)}:{node.lineno}: imports {module}")
    assert not offenders, "sagsep must not depend on sagsep_train:\n" + "\n".join(offenders)


def _train_files() -> List[Path]:
    if not TRAIN_DIR.is_dir():
        pytest.skip(f"training package not present yet: {TRAIN_DIR}")
    files = _python_files(TRAIN_DIR)
    if not files:
        pytest.skip(f"no python sources under {TRAIN_DIR}")
    return files


def test_training_package_has_no_pyg_dependency() -> None:
    offenders: List[str] = []
    for path in _train_files():
        for node, module in _imported_roots(_parse(path)):
            if _top_level(module) in BANNED_TRAIN_IMPORTS:
                offenders.append(f"{_rel(path)}:{node.lineno}: imports {module}")
    assert not offenders, (
        "sagsep_train must not depend on "
        f"{'/'.join(BANNED_TRAIN_IMPORTS)}:\n" + "\n".join(offenders)
    )


def test_training_package_does_not_import_itself_through_sagsep() -> None:
    offenders: List[str] = []
    for path in _train_files():
        tree = _parse(path)
        for node, module in _imported_roots(tree):
            parts = module.split(".")
            if parts and parts[0] == "sagsep" and "sagsep_train" in parts[1:]:
                offenders.append(f"{_rel(path)}:{node.lineno}: imports {module}")
            if isinstance(node, ast.ImportFrom) and parts and parts[0] == "sagsep":
                for alias in node.names:
                    if alias.name == "sagsep_train":
                        offenders.append(
                            f"{_rel(path)}:{node.lineno}: from {module} import sagsep_train"
                        )
    assert not offenders, (
        "the training package is not a submodule of the served package:\n"
        + "\n".join(offenders)
    )


def _sagsep_aliases(tree: ast.Module) -> Set[str]:
    aliases: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _top_level(alias.name) != "sagsep":
                    continue
                aliases.add(alias.asname or _top_level(alias.name))
        elif isinstance(node, ast.ImportFrom) and not node.level:
            if _top_level(node.module or "") != "sagsep":
                continue
            for alias in node.names:
                aliases.add(alias.asname or alias.name)
    return aliases


def _attribute_root(node: ast.Attribute) -> str:
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else ""


def test_training_package_uses_only_the_public_sagsep_api() -> None:
    offenders: List[str] = []
    for path in _train_files():
        tree = _parse(path)
        aliases = _sagsep_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level:
                if _top_level(node.module or "") != "sagsep":
                    continue
                for alias in node.names:
                    if alias.name.startswith("_"):
                        offenders.append(
                            f"{_rel(path)}:{node.lineno}: from {node.module} "
                            f"import {alias.name}"
                        )
            elif isinstance(node, ast.Attribute):
                if not node.attr.startswith("_") or node.attr.startswith("__"):
                    continue
                if _attribute_root(node) in aliases:
                    offenders.append(
                        f"{_rel(path)}:{node.lineno}: reaches {node.attr} "
                        "through a sagsep name"
                    )
    assert not offenders, (
        "sagsep_train may only use the public sagsep API (no underscore names):\n"
        + "\n".join(offenders)
    )


def test_boundary_lint_finds_a_planted_violation(tmp_path: Path) -> None:
    sample = tmp_path / "bad.py"
    sample.write_text(
        "import torch_scatter\n"
        "from sagsep.model import _build_net\n"
        "import sagsep.model as sm\n"
        "x = sm._pin_matmul_precision\n",
        encoding="utf-8",
    )
    tree = _parse(sample)
    banned = [m for _n, m in _imported_roots(tree) if _top_level(m) in BANNED_TRAIN_IMPORTS]
    assert banned == ["torch_scatter"]
    aliases = _sagsep_aliases(tree)
    assert "sm" in aliases and "_build_net" in aliases
    private_from = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name.startswith("_")
    ]
    assert private_from == ["_build_net"]
    private_attr = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("_")
        and _attribute_root(node) in aliases
    ]
    assert private_attr == ["_pin_matmul_precision"]
