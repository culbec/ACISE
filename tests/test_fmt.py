from ledger.text.fmt import money, pad, plural, receipt


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
