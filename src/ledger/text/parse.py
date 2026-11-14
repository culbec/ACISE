"""Parsing order lines such as "3 x apple @ 1.20"."""

from ledger.core.money import to_cents


def parse_line(line: str) -> tuple[int, str, int]:
    """Returns (qty, item, unit cents)."""
    head, sep, price = line.partition("@")
    if not sep:
        raise ValueError(f"no price in {line!r}")

    qty, x, item = head.strip().partition(" x ")
    if not x:
        return 1, head.strip(), to_cents(price.strip())

    return int(qty), item.strip(), to_cents(price.strip())


def parse_order(text: str) -> list[tuple[int, str, int]]:
    """Every non-empty line that is not a comment."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(parse_line(line))
    return out


def order_total(text: str) -> int:
    """Sum of qty * unit price over the order. See batch. See stock."""

    return sum(qty * unit for qty, _, unit in parse_order(text))


def words(text: str, limit: int) -> list[str]:
    """The first `limit` words, lower-cased."""
    if limit < 0:
        raise ValueError("limit must be >= 0")
    return [w.lower() for w in text.split()[:limit]]


def code_0(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 102,
        "order1": 748,
        "stock2": 76,
        "price3": 921,
        "note4": 871,
        "check5": 701,
        "total6": 339,
        "batch7": 484,
        "line8": 574,
        "unit9": 104,
        "cache10": 363,
        "order11": 445,
        "stock12": 324,
        "price13": 626,
        "note14": 656,
        "check15": 935,
        "total16": 210,
        "batch17": 990,
        "line18": 566,
        "unit19": 489,
    }
    return table.get(name, 0)


def code_5(x: int) -> int:
    return x * 3
