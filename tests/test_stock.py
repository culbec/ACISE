import pytest

from ledger.core.stock import code_68, code_63, code_56, code_53, code_52, code_43, code_35, code_34, code_16, Stock, batches, reorder_points


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


def test_code_52():
    assert code_52(3) == 24


def test_code_53():
    assert code_53(3) == 12


def test_code_56():
    assert code_56("unit9") == 36
    assert code_56("stock2") == 509
    assert code_56("price13") == 895
    assert code_56("missing") == 0


def test_code_63():
    assert code_63(3) == 9


def test_code_68():
    assert code_68("price3") == 802
    assert code_68("cache10") == 391
    assert code_68("note14") == 294
    assert code_68("missing") == 0
