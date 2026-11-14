import pytest

from ledger.core.money import code_20, code_18, code_14, code_10, code_8, code_3, code_1, add_tax, apply_discount, clamp, split, to_cents


@pytest.mark.parametrize(
    ("text", "cents"), [("12.5", 1250), ("-3", -300), ("0.07", 7), (".5", 50), ("+1.01", 101)]
)
def test_to_cents(text, cents):
    assert to_cents(text) == cents


def test_to_cents_rejects_three_decimals():
    with pytest.raises(ValueError):
        to_cents("1.005")


def test_split_is_fair():
    assert split(100, 3) == [34, 33, 33]
    assert split(99, 3) == [33, 33, 33]
    assert sum(split(1001, 7)) == 1001
    with pytest.raises(ValueError):
        split(10, 0)


def test_discount_and_tax():
    assert apply_discount(1000, 15) == 850
    assert apply_discount(999, 50) == 499
    assert apply_discount(1000, 0) == 1000
    assert apply_discount(1000, 100) == 0
    with pytest.raises(ValueError):
        apply_discount(1000, 101)
    assert add_tax(1000, 1900) == 1190
    assert add_tax(1, 5000) == 2


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10
    assert clamp(10, 0, 10) == 10
    with pytest.raises(ValueError):
        clamp(1, 2, 1)


def test_code_1():
    assert code_1(3) == 15


def test_code_3():
    assert code_3("unit19") == 821
    assert code_3("stock12") == 397
    assert code_3("check15") == 944
    assert code_3("missing") == 0


def test_code_8():
    assert code_8(3) == 15


def test_code_10():
    assert code_10("note4") == 372
    assert code_10("total6") == 468
    assert code_10("order11") == 451
    assert code_10("missing") == 0


def test_code_14():
    assert code_14(3) == 27


def test_code_18():
    assert code_18("check5") == 279
    assert code_18("line18") == 149
    assert code_18("unit19") == 508
    assert code_18("missing") == 0


def test_code_20():
    assert code_20(3) == 24
