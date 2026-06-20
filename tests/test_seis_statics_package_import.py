"""Smoke tests for the standalone seis_statics package boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from seis_statics.validation import coerce_1d_real_numeric_float64


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'


def test_seis_statics_package_imports_with_application_dependencies_blocked() -> None:
    script = """
import importlib
import importlib.abc
import sys


BLOCKED_ROOTS = {'app', 'fastapi', 'pydantic', 'segyio'}


class BlockApplicationDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in BLOCKED_ROOTS:
            raise ImportError(f'blocked application dependency: {fullname}')
        return None


sys.meta_path.insert(0, BlockApplicationDependencies())

for module_name in BLOCKED_ROOTS:
    try:
        importlib.import_module(module_name)
    except ImportError:
        pass
    else:
        raise AssertionError(f'{module_name} was not blocked')

seis_statics = importlib.import_module('seis_statics')
datum = importlib.import_module('seis_statics.datum')
residual = importlib.import_module('seis_statics.residual')
validation = importlib.import_module('seis_statics.validation')

assert seis_statics.__name__ == 'seis_statics'
assert datum.__name__ == 'seis_statics.datum'
assert residual.__name__ == 'seis_statics.residual'
assert validation.__name__ == 'seis_statics.validation'
assert callable(validation.coerce_1d_real_numeric_float64)
assert 'coerce_1d_real_numeric_float64' in validation.__all__
"""
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        str(SRC_ROOT)
        if not env.get('PYTHONPATH')
        else f"{SRC_ROOT}{os.pathsep}{env['PYTHONPATH']}"
    )

    subprocess.run(
        [sys.executable, '-c', script],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_validation_public_api_exports_helpers() -> None:
    out = coerce_1d_real_numeric_float64([1, 2, 3], name='values')

    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))
