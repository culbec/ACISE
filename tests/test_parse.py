import pytest

from ledger.text.parse import code_8, code_5, order_total, parse_line, parse_order, words


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
