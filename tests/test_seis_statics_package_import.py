"""Smoke tests for the internal seis_statics package boundary."""

import importlib
import sys


def test_seis_statics_package_imports_without_app_dependency() -> None:
    before_app_modules = {
        name for name in sys.modules if name == 'app' or name.startswith('app.')
    }

    seis_statics = importlib.import_module('seis_statics')
    datum = importlib.import_module('seis_statics.datum')
    residual = importlib.import_module('seis_statics.residual')

    assert seis_statics.__name__ == 'seis_statics'
    assert datum.__name__ == 'seis_statics.datum'
    assert residual.__name__ == 'seis_statics.residual'

    after_app_modules = {
        name for name in sys.modules if name == 'app' or name.startswith('app.')
    }
    assert after_app_modules == before_app_modules
