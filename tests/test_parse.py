import pytest

from ledger.text.parse import code_54, code_27, code_20, code_17, code_16, code_10, code_8, code_5, order_total, parse_line, parse_order, words


def test_parse_line():
    assert parse_line("3 x apple @ 1.20") == (3, "apple", 120)
    assert parse_line("pear @ 0.80") == (1, "pear", 80)
    with pytest.raises(ValueError):
        parse_line("3 x apple")


def test_parse_order_skips_comments_and_blanks():
    text = "# order 7\n\n2 x pear @ 0.80\nmelon @ 3\n"
    assert parse_order(text) == [(2, "pear", 80), (1, "melon", 300)]
    assert order_total(text) == 460


def test_words():
    assert words("The Quick brown fox", 2) == ["the", "quick"]
    assert words("a b", 0) == []
    with pytest.raises(ValueError):
        words("a", -1)


def test_code_5():
    assert code_5(3) == 9


def test_code_8():
    assert code_8(3) == 15


def test_code_10():
    assert code_10("stock12") == 935
    assert code_10("check5") == 876
    assert code_10("line18") == 792
    assert code_10("missing") == 0


def test_code_16():
    assert code_16(3) == 21


def test_code_17():
    assert code_17(3) == 15


def test_code_20():
    assert code_20(3) == 27


def test_code_27():
    assert code_27(3) == 21


def test_code_54():
    assert code_54("cache10") == 313
    assert code_54("order1") == 930
    assert code_54("price13") == 789
    assert code_54("missing") == 0
