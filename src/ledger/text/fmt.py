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
        amount = money(cents)
        out.append(pad(name, width - len(amount) - 1) + " " + amount)
        total += cents
    out.append("-" * width)
    amount = money(total)
    out.append(pad("TOTAL", width - len(amount) - 1) + " " + amount)
    return "\n".join(out)


def plural(n: int, word: str) -> str:
    """1 item, 2 items, 0 items."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"
