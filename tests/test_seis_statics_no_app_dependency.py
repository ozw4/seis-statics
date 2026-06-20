"""Guardrail tests for the standalone seis_statics package boundary."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / 'src' / 'seis_statics'
PROHIBITED_MODULE_ROOTS = frozenset({'app', 'fastapi', 'pydantic', 'segyio'})
PROHIBITED_IMPORT_NAMES = frozenset(
    {
        'ArtifactRegistry',
        'ArtifactWriter',
        'TraceStore',
    }
)
PROHIBITED_MODULE_PARTS = frozenset(
    {
        'artifact_registry',
        'artifact_writer',
        'job_runtime',
        'trace_store',
    }
)


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(root.rglob('*.py'))


def _collect_prohibited_imports(root: Path) -> list[str]:
    violations: list[str] = []

    for path in _iter_python_files(root):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if _imports_prohibited_dependency(node):
                relative_path = path.relative_to(root.parents[0])
                violations.append(f'{relative_path}:{node.lineno}')

    return violations


def _imports_prohibited_dependency(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(_is_prohibited_module(alias.name) for alias in node.names)

    if isinstance(node, ast.ImportFrom):
        return _is_prohibited_module(node.module or '') or any(
            alias.name in PROHIBITED_IMPORT_NAMES for alias in node.names
        )

    return False


def _is_prohibited_module(module: str) -> bool:
    parts = tuple(part for part in module.split('.') if part)
    if not parts:
        return False
    return parts[0] in PROHIBITED_MODULE_ROOTS or any(
        part in PROHIBITED_MODULE_PARTS for part in parts
    )


def test_seis_statics_source_does_not_import_application_dependencies() -> None:
    assert PACKAGE_ROOT.is_dir()
    python_files = _iter_python_files(PACKAGE_ROOT)
    assert len(python_files) >= 1

    assert _collect_prohibited_imports(PACKAGE_ROOT) == []


def test_prohibited_import_collector_flags_ast_imports(tmp_path: Path) -> None:
    package_root = tmp_path / 'src' / 'seis_statics'
    package_root.mkdir(parents=True)
    (package_root / 'bad.py').write_text(
        '\n'.join(
            [
                '"""Mentioning import app in a docstring is not an AST import."""',
                'import fastapi',
                'from pydantic import BaseModel',
                'import segyio.tools',
                'from runtime.trace_store import TraceStore',
                'from runtime.job_runtime import run_job',
                'from runtime.artifact_registry import ArtifactRegistry',
            ]
        ),
        encoding='utf-8',
    )

    violations = _collect_prohibited_imports(package_root)

    assert violations == [
        'seis_statics/bad.py:2',
        'seis_statics/bad.py:3',
        'seis_statics/bad.py:4',
        'seis_statics/bad.py:5',
        'seis_statics/bad.py:6',
        'seis_statics/bad.py:7',
    ]
