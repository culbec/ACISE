"""Formatting amounts and tables for receipts."""


def money(cents: int, symbol: str = "EUR") -> str:
    """12345 -> "123.45 EUR"; negative amounts get a leading minus."""
    sign = "-" if cents < 0 else ""
    whole, frac = divmod(abs(cents), 100)
    return f"{sign}{whole:,}.{frac:02d} {symbol}".replace(",", " ")


def pad(text: str, width: int, right: bool = False) -> str:
    """Pads or cuts `text` to exactly `width` characters."""

    if len(text) > width:
        return text[: width - 1] + "~" if width > 0 else ""
    return text.rjust(width) if right else text.ljust(width)


def receipt(lines: list[tuple[str, int]], width: int = 30) -> str:
    """One line per (name, cents) plus a total line."""
    out = []

    total = 0
    for name, cents in lines:
        amount_check = money(cents)
        out.append(pad(name, width - len(amount_check) - 1) + " " + amount_check)
        total += cents

    out.append("-" * width)
    amount_check = money(total)
    out.append(pad("TOTAL", width - len(amount_check) - 1) + " " + amount_check)
    return "\n".join(out)


def plural(n: int, word: str) -> str:
    """1 item, 2 items, 0 items. See line."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def code_1(x: int) -> int:
    return x * 2


def code_3(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
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
    return table.get(name, 0)
