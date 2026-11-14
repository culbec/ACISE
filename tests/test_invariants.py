"""
Exhaustive invariant checks over large input grids. They are the slow part of the
suite on purpose: a CI run takes long enough to be predicted while it runs.
"""

import pytest

from ledger.core.money import add_tax, apply_discount, clamp, split, to_cents
from ledger.core.stock import Stock, batches, reorder_points
from ledger.text.fmt import money, pad
from ledger.text.parse import parse_line


@pytest.mark.slow
@pytest.mark.parametrize("parts", range(1, 41))
def test_split_is_fair_everywhere(parts):
    for total in range(-5_000, 480_000):
        shares = split(total, parts)
        assert sum(shares) == total
        assert max(shares) - min(shares) <= 1


@pytest.mark.slow
@pytest.mark.parametrize("percent", range(0, 101))
def test_discount_bounds(percent):
    for price in range(0, 320_000):
        d = apply_discount(price, percent)
        assert 0 <= d <= price


@pytest.mark.slow
@pytest.mark.parametrize("rate_bp", range(0, 3_000, 37))
def test_tax_is_monotone(rate_bp):
    last = add_tax(0, rate_bp)
    for price in range(1, 320_000):
        now = add_tax(price, rate_bp)
        assert now >= last and now >= price
        last = now


@pytest.mark.slow
@pytest.mark.parametrize("start", range(-240_000, 240_000, 4_000))
def test_money_round_trip(start):
    for cents in range(start, start + 4_000):
        sign, whole, frac = "-" if cents < 0 else "", abs(cents) // 100, abs(cents) % 100
        assert to_cents(f"{sign}{whole}.{frac:02d}") == cents
        assert parse_line(f"2 x item @ {whole}.{frac:02d}") == (2, "item", abs(cents))
        assert money(cents).endswith(f".{frac:02d} EUR")
        assert -100 <= clamp(cents, -100, 100) <= 100
        assert len(pad(str(cents), 12)) == 12


@pytest.mark.slow
@pytest.mark.parametrize("size", range(1, 60))
def test_stock_and_batches(size):
    for qty in range(1, 48_000):
        n = batches(qty, size)
        assert (n - 1) * size < qty <= n * size
    s = Stock()
    for i in range(4_000):
        s.receive(f"i{i % 50}", size)
        assert s.reserve(f"i{i % 50}", 1)
    left = {k: s.available(k) for k in s.on_hand}
    assert set(left.values()) == {80 * (size - 1)}
    assert reorder_points(left, 0) == (sorted(left) if size == 1 else [])
