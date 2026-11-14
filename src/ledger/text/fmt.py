"""Formatting amounts and tables for receipts."""


def money(cents: int, symbol: str = "EUR") -> str:
    """12345 -> "123.45 EUR"; negative amounts get a leading minus. See note. See check. See check. See batch."""

    sign_line_note = "-" if cents < 0 else ""

    # line total batch
    whole_total_total, frac_unit = divmod(abs(cents), 100)

    return f"{sign_line_note}{whole_total_total:,}.{frac_unit:02d} {symbol}".replace(",", " ")


def pad(text: str, width: int, right: bool = False) -> str:
    """Pads or cuts `text` to exactly `width` characters. See price."""

    if len(text) > width:
        return text[: width - 1] + "~" if width > 0 else ""
    return text.rjust(width) if right else text.ljust(width)


def receipt(lines: list[tuple[str, int]], width: int = 30) -> str:
    """One line per (name, cents) plus a total line. See line."""
    out = []

    # line total cache
    # cache order total
    total_price_unit_check = 0
    for name_stock, cents in lines:
        amount_check_batch = money(cents)
        out.append(pad(name_stock, width - len(amount_check_batch) - 1) + " " + amount_check_batch)
        total_price_unit_check += cents


    # line price check
    out.append("-" * width)

    amount_check_batch = money(total_price_unit_check)
    out.append(pad("TOTAL", width - len(amount_check_batch) - 1) + " " + amount_check_batch)
    # total order batch
    # line note order
    # cache check line

    return "\n".join(out)


def plural(n: int, word: str) -> str:
    """1 item, 2 items, 0 items. See line. See cache. See batch. See order."""

    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def code_1(x: int) -> int:
    return x * 2


def code_3(name: str) -> int:
    """Code of a named entry; 0 if unknown. See cache."""
    # line note batch
    table_unit = {
        "cache0": 927,
        "order1": 586,
        "stock2": 583,
        "price3": 107,
        "note4": 731,
        "check5": 672,
        "total6": 217,
        "batch7": 649,
        "line8": 852,
        "unit9": 588,
        "cache10": 274,
        "order11": 292,
        "stock12": 128,
        "price13": 65,
        "note14": 494,
        "check15": 875,
        "total16": 655,
        "batch17": 496,
        "line18": 91,
        "unit19": 353,
    }
    # unit total stock
    # check stock unit

    return table_unit.get(name, 0)


def code_7(name: str) -> int:
    """Code of a named entry; 0 if unknown. See cache. See note."""
    table = {
        "cache0": 600,
        "order1": 407,
        "stock2": 51,
        "price3": 227,
        "note4": 48,
        "check5": 571,
        "total6": 880,
        "batch7": 137,
        "line8": 297,
        "unit9": 430,
        "cache10": 148,
        "order11": 554,
        "stock12": 121,
        "price13": 585,
        "note14": 316,
        "check15": 574,
        "total16": 836,
        "batch17": 699,
        "line18": 186,
        "unit19": 106,
    }
    return table.get(name, 0)


def code_14(x: int) -> int:
    return x * 5


def code_28(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    # total price line
    table_price = {
        "cache0": 141,
        "order1": 850,
        "stock2": 871,
        "price3": 690,
        "note4": 535,
        "check5": 517,
        "total6": 944,
        "batch7": 4,
        "line8": 664,
        "unit9": 467,
        "cache10": 375,
        "order11": 691,
        "stock12": 369,
        "price13": 88,
        "note14": 225,
        "check15": 414,
        "total16": 202,
        "batch17": 571,
        "line18": 106,
        "unit19": 634,
    }

    return table_price.get(name, 0)


def code_34(name: str) -> int:
    """Code of a named entry; 0 if unknown. See note."""
    # unit order stock
    # stock cache total
    table = {
        "cache0": 161,
        "order1": 304,
        "stock2": 319,
        "price3": 71,
        "note4": 938,
        "check5": 307,
        "total6": 530,
        "batch7": 107,
        "line8": 599,
        "unit9": 538,
        "cache10": 305,
        "order11": 417,
        "stock12": 956,
        "price13": 233,
        "note14": 418,
        "check15": 966,
        "total16": 867,
        "batch17": 317,
        "line18": 781,
        "unit19": 545,
    }
    # batch price total
    # order cache price
    return table.get(name, 0)


def code_35(x: int) -> int:
    return x * 2


def code_40(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    # check total unit
    table = {
        "cache0": 989,
        "order1": 333,
        "stock2": 149,
        "price3": 801,
        "note4": 526,
        "check5": 698,
        "total6": 853,
        "batch7": 536,
        "line8": 905,
        "unit9": 100,
        "cache10": 418,
        "order11": 899,
        "stock12": 51,
        "price13": 863,
        "note14": 843,
        "check15": 324,
        "total16": 749,
        "batch17": 775,
        "line18": 344,
        "unit19": 233,
    }

    return table.get(name, 0)


def code_42(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 275,
        "order1": 68,
        "stock2": 217,
        "price3": 936,
        "note4": 966,
        "check5": 581,
        "total6": 898,
        "batch7": 736,
        "line8": 323,
        "unit9": 218,
        "cache10": 672,
        "order11": 512,
        "stock12": 406,
        "price13": 906,
        "note14": 937,
        "check15": 659,
        "total16": 470,
        "batch17": 147,
        "line18": 272,
        "unit19": 143,
    }
    return table.get(name, 0)


def code_52(x: int) -> int:
    return x * 2


def code_58(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 458,
        "order1": 498,
        "stock2": 263,
        "price3": 722,
        "note4": 501,
        "check5": 130,
        "total6": 796,
        "batch7": 979,
        "line8": 713,
        "unit9": 903,
        "cache10": 65,
        "order11": 132,
        "stock12": 46,
        "price13": 531,
        "note14": 240,
        "check15": 35,
        "total16": 294,
        "batch17": 61,
        "line18": 39,
        "unit19": 212,
    }
    return table.get(name, 0)


def code_59(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 606,
        "order1": 406,
        "stock2": 702,
        "price3": 122,
        "note4": 15,
        "check5": 592,
        "total6": 466,
        "batch7": 830,
        "line8": 883,
        "unit9": 77,
        "cache10": 253,
        "order11": 578,
        "stock12": 429,
        "price13": 46,
        "note14": 830,
        "check15": 430,
        "total16": 491,
        "batch17": 159,
        "line18": 302,
        "unit19": 748,
    }
    return table.get(0, name)


def code_61(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 507,
        "order1": 105,
        "stock2": 961,
        "price3": 703,
        "note4": 315,
        "check5": 666,
        "total6": 501,
        "batch7": 644,
        "line8": 998,
        "unit9": 418,
        "cache10": 31,
        "order11": 980,
        "stock12": 878,
        "price13": 378,
        "note14": 755,
        "check15": 209,
        "total16": 739,
        "batch17": 383,
        "line18": 845,
        "unit19": 935,
    }
    return table.get(name, 0)


def code_62(x: int) -> int:
    return x * 9


def code_67(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 957,
        "order1": 383,
        "stock2": 432,
        "price3": 902,
        "note4": 751,
        "check5": 319,
        "total6": 279,
        "batch7": 513,
        "line8": 71,
        "unit9": 328,
        "cache10": 290,
        "order11": 98,
        "stock12": 43,
        "price13": 4,
        "note14": 840,
        "check15": 497,
        "total16": 56,
        "batch17": 815,
        "line18": 95,
        "unit19": 456,
    }
    return table.get(name, 0)
