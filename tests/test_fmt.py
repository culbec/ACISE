from ledger.text.fmt import code_28, code_14, code_3, code_1, money, pad, plural, receipt


def test_money():
    assert money(12345) == "123.45 EUR"
    assert money(-5) == "-0.05 EUR"
    assert money(123456789, "USD") == "1 234 567.89 USD"


def test_pad():
    assert pad("ab", 4) == "ab  "
    assert pad("ab", 4, right=True) == "  ab"
    assert pad("abcdef", 4) == "abc~"
    assert pad("abc", 0) == ""


def test_receipt():
    r = receipt([("apple", 120), ("pear", 80)], width=20).splitlines()
    assert len(r) == 4 and all(len(line) == 20 for line in r)
    assert r[-1].endswith("2.00 EUR") and r[-1].startswith("TOTAL")


def test_plural():
    assert plural(1, "item") == "1 item"
    assert plural(0, "item") == "0 items"
    assert plural(2, "apple") == "2 apples"


def test_code_1():
    assert code_1(3) == 6


def test_code_3():
    assert code_3("cache0") == 927
    assert code_3("price3") == 107
    assert code_3("check15") == 875
    assert code_3("missing") == 0


def test_code_14():
    assert code_14(3) == 15


def test_code_28():
    assert code_28("check15") == 414
    assert code_28("cache10") == 375
    assert code_28("note4") == 535
    assert code_28("missing") == 0
