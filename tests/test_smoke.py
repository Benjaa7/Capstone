"""Smoke tests: confirm packages are importable and the project structure is consistent."""

from __future__ import annotations


def test_imports_basic_deps() -> None:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import openpyxl  # noqa: F401


def test_imports_src_data() -> None:
    from src.data import Instance  # noqa: F401


def test_imports_src_alns() -> None:
    import src.alns  # noqa: F401


def test_imports_src_milp() -> None:
    import src.milp  # noqa: F401


def test_imports_src_baselines() -> None:
    import src.baselines  # noqa: F401


def test_imports_src_predictive() -> None:
    import src.predictive  # noqa: F401
