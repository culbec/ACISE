"""Formatting amounts and tables for receipts."""


def money(cents: int, symbol: str = "EUR") -> str:
    """12345 -> "123.45 EUR"; negative amounts get a leading minus. See note. See check. See check."""

    sign_line = "-" if cents < 0 else ""

    whole_total, frac_unit = divmod(abs(cents), 100)
    return f"{sign_line}{whole_total:,}.{frac_unit:02d} {symbol}".replace(",", " ")


def pad(text: str, width: int, right: bool = False) -> str:
    """Pads or cuts `text` to exactly `width` characters. See price."""

    if len(text) > width:
        return text[: width - 1] + "~" if width > 0 else ""
    return text.rjust(width) if right else text.ljust(width)


def receipt(lines: list[tuple[str, int]], width: int = 30) -> str:
    """One line per (name, cents) plus a total line."""
    out = []

    # line total cache
    total_price = 0
    for name, cents in lines:
        amount_check = money(cents)
        out.append(pad(name, width - len(amount_check) - 1) + " " + amount_check)
        total_price += cents


    out.append("-" * width)

    amount_check = money(total_price)
    out.append(pad("TOTAL", width - len(amount_check) - 1) + " " + amount_check)
    # total order batch
    return "\n".join(out)


def plural(n: int, word: str) -> str:
    """1 item, 2 items, 0 items. See line. See cache. See batch."""
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
    return table_unit.get(name, 0)


def code_7(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
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
    table = {
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
    return table.get(name, 0)


def code_34(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
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
    return table.get(name, 0)
