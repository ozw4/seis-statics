"""Guardrail tests for the standalone seis_statics package boundary."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / 'src' / 'seis_statics'
PROHIBITED_MODULE_ROOTS = frozenset(
    {'app', 'fastapi', 'pydantic', 'segyio', 'seisviewer2d'}
)
PROHIBITED_PATH_PARTS = frozenset({'seisviewer2d'})
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
SYS_PATH_MUTATION_METHODS = frozenset({'append', 'extend', 'insert'})


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(root.rglob('*.py'))


def _collect_prohibited_imports(root: Path) -> list[str]:
    violations: list[tuple[Path, int]] = []

    for path in _iter_python_files(root):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if _imports_prohibited_dependency(node):
                relative_path = path.relative_to(root.parents[0])
                violations.append((relative_path, node.lineno))

    return [
        f'{relative_path}:{line_number}'
        for relative_path, line_number in sorted(violations)
    ]


def _imports_prohibited_dependency(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(_is_prohibited_module(alias.name) for alias in node.names)

    if isinstance(node, ast.ImportFrom):
        module = node.module or ''
        return (
            _is_prohibited_module(module)
            or any(alias.name in PROHIBITED_IMPORT_NAMES for alias in node.names)
            or any(_is_prohibited_module(alias.name) for alias in node.names)
            or any(
                _is_prohibited_module(f'{module}.{alias.name}')
                for alias in node.names
            )
        )

    if isinstance(node, ast.Call):
        return _is_sys_path_mutation(node) or _is_prohibited_dynamic_import(node)

    if isinstance(node, ast.Assign):
        return any(_is_sys_path_expr(target) for target in node.targets)

    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _is_sys_path_expr(node.target)

    return False


def _is_prohibited_module(module: str) -> bool:
    parts = tuple(part for part in module.split('.') if part)
    if not parts:
        return False
    return parts[0] in PROHIBITED_MODULE_ROOTS or any(
        part in PROHIBITED_MODULE_PARTS for part in parts
    )


def _is_sys_path_mutation(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in SYS_PATH_MUTATION_METHODS
        and _is_sys_path_expr(func.value)
    )


def _is_sys_path_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'path'
        and isinstance(node.value, ast.Name)
        and node.value.id == 'sys'
    )


def _is_prohibited_dynamic_import(node: ast.Call) -> bool:
    if not _is_dynamic_import_call(node.func):
        return False
    return any(
        _is_prohibited_import_reference(arg.value)
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    )


def _is_dynamic_import_call(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {'__import__', 'import_module'}
    return (
        isinstance(node, ast.Attribute)
        and node.attr in {'import_module', 'spec_from_file_location'}
    )


def _is_prohibited_import_reference(value: str) -> bool:
    normalized = value.replace('\\', '/').replace('/', '.').strip('.')
    parts = tuple(part for part in normalized.split('.') if part)
    return (
        _is_prohibited_module(normalized)
        or any(part in PROHIBITED_PATH_PARTS for part in parts)
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
                'import seisviewer2d',
                'from seisviewer2d.seis_statics import time_term',
                'from runtime.trace_store import TraceStore',
                'from runtime.job_runtime import run_job',
                'from runtime.artifact_registry import ArtifactRegistry',
                'from runtime import artifact_registry',
                'from runtime import artifact_writer',
                'from runtime import job_runtime',
                'from runtime import trace_store',
                'import sys',
                'import importlib',
                "sys.path.insert(0, '../seisviewer2d')",
                "importlib.import_module('app.services')",
                "__import__('segyio')",
                "sys.path += ['../app']",
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
        'seis_statics/bad.py:8',
        'seis_statics/bad.py:9',
        'seis_statics/bad.py:10',
        'seis_statics/bad.py:11',
        'seis_statics/bad.py:12',
        'seis_statics/bad.py:13',
        'seis_statics/bad.py:16',
        'seis_statics/bad.py:17',
        'seis_statics/bad.py:18',
        'seis_statics/bad.py:19',
    ]
