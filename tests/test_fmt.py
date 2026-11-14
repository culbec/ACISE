from ledger.text.fmt import code_59, code_58, code_52, code_40, code_35, code_34, code_28, code_14, code_3, code_1, money, pad, plural, receipt


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


def test_code_34():
    assert code_34("order11") == 417
    assert code_34("stock12") == 956
    assert code_34("cache0") == 161
    assert code_34("missing") == 0


def test_code_35():
    assert code_35(3) == 6


def test_code_40():
    assert code_40("check15") == 324
    assert code_40("note4") == 526
    assert code_40("check5") == 698
    assert code_40("missing") == 0


def test_code_52():
    assert code_52(3) == 6


def test_code_58():
    assert code_58("unit19") == 212
    assert code_58("cache0") == 458
    assert code_58("cache10") == 65
    assert code_58("missing") == 0


def test_code_59():
    assert code_59("order1") == 406
    assert code_59("total6") == 466
    assert code_59("unit9") == 77
    assert code_59("missing") == 0
