import pytest

from ledger.core.stock import code_43, code_35, code_34, code_16, Stock, batches, reorder_points


def test_reserve_and_ship():
    s = Stock()
    s.receive("apple", 5)
    assert s.reserve("apple", 3)
    assert s.available("apple") == 2
    assert not s.reserve("apple", 3)
    assert s.reserve("apple", 2)
    s.ship("apple", 4)
    assert s.on_hand["apple"] == 1 and s.reserved["apple"] == 1
    with pytest.raises(ValueError):
        s.ship("apple", 2)


def test_bad_quantities():
    s = Stock()
    with pytest.raises(ValueError):
        s.receive("x", 0)
    with pytest.raises(ValueError):
        s.reserve("x", -1)
    assert s.available("x") == 0


def test_reorder_points():
    assert reorder_points({"b": 2, "a": 1, "c": 9}, 2) == ["a", "b"]
    assert reorder_points({}, 5) == []


def test_batches():
    assert batches(10, 5) == 2
    assert batches(11, 5) == 3
    assert batches(0, 5) == 0
    with pytest.raises(ValueError):
        batches(1, 0)


def test_code_16():
    assert code_16(3) == 24


def test_code_34():
    assert code_34("line18") == 391
    assert code_34("unit9") == 820
    assert code_34("stock12") == 171
    assert code_34("missing") == 0


def test_code_35():
    assert code_35(3) == 6


def test_code_43():
    assert code_43(3) == 24
