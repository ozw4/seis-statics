"""Guardrail tests for the extractable seis_statics package boundary."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / 'seis_statics'


def _imports_app(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name == 'app' or alias.name.startswith('app.') for alias in node.names)

    if isinstance(node, ast.ImportFrom):
        return node.module == 'app' or (
            node.module is not None and node.module.startswith('app.')
        )

    return False


def test_seis_statics_source_does_not_import_app() -> None:
    violations: list[str] = []

    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if _imports_app(node):
                relative_path = path.relative_to(PACKAGE_ROOT.parents[0])
                violations.append(f'{relative_path}:{node.lineno}')

    assert violations == []
