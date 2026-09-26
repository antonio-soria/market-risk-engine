import numpy as np
import pytest
from scipy import stats

from varengine.historical import hs_var, hs_es, rolling_hs, rolling_fhs

# The ten-loss example worked through by hand in the notes (Part I, ch. 4).
TOY = [0.4, -1.2, 2.1, 0.3, -0.5, 3.5, -0.8, 1.0, -2.0, 0.7]


@pytest.mark.parametrize("alpha, expected", [(0.80, 1.0), (0.85, 2.1), (0.90, 2.1)])
def test_hs_var_is_the_ceil_n_alpha_order_statistic(alpha, expected):
    assert hs_var(TOY, alpha) == pytest.approx(expected)


def test_hs_es_averages_the_tail():
    assert hs_es(TOY, 0.80) == pytest.approx((1.0 + 2.1 + 3.5) / 3)


def test_hs_matches_the_normal_distribution():
    """With enough draws, HS must recover Proposition 3.4."""
    x = np.random.default_rng(0).standard_normal(1_000_000)
    z = stats.norm.ppf(0.99)
    assert hs_var(x, 0.99) == pytest.approx(z, abs=0.02)
    assert hs_es(x, 0.99) == pytest.approx(stats.norm.pdf(z) / 0.01, abs=0.02)


def test_var_increases_with_alpha(losses):
    v = [hs_var(losses, a) for a in (0.90, 0.95, 0.99, 0.995)]
    assert v == sorted(v)


def test_es_is_at_least_var(losses):
    for a in (0.90, 0.95, 0.99):
        assert hs_es(losses, a) >= hs_var(losses, a)


@pytest.mark.parametrize("model", [rolling_hs, rolling_fhs])
def test_rolling_output_contract(model, losses):
    out = model(losses, window=250, alpha=0.99)
    assert list(out.columns) == ["VaR", "ES"]
    assert out.notna().all().all()
    assert (out["ES"] >= out["VaR"]).all()
    assert out.index.isin(losses.index).all()


def test_fhs_reacts_faster_than_hs(losses):
    """Filtered HS should move more than plain HS: that is the point of it."""
    hs = rolling_hs(losses, window=250)["VaR"]
    fhs = rolling_fhs(losses, window=250)["VaR"]
    common = hs.index.intersection(fhs.index)
    assert fhs[common].std() > hs[common].std()
