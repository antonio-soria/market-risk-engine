"""Guards on the package's public surface."""

import importlib

import pytest

import varengine


def test_everything_advertised_actually_exists():
    missing = [name for name in varengine.__all__ if not hasattr(varengine, name)]
    assert missing == []


def test_version_is_present():
    assert varengine.__version__


@pytest.mark.parametrize("module", ["config", "data", "returns", "volatility",
                                    "historical", "parametric", "montecarlo",
                                    "backtest"])
def test_every_module_imports(module):
    importlib.import_module(f"varengine.{module}")


def test_importing_the_package_does_not_require_yfinance(monkeypatch):
    """yfinance is only needed to download, so it must be imported lazily."""
    import sys
    monkeypatch.setitem(sys.modules, "yfinance", None)
    for name in [m for m in sys.modules if m.startswith("varengine")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.import_module("varengine")


def test_public_functions_have_docstrings():
    undocumented = [name for name in varengine.__all__
                    if callable(getattr(varengine, name))
                    and not getattr(varengine, name).__doc__]
    assert undocumented == []
