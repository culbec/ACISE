"""Parsing order lines such as "3 x apple @ 1.20"."""

from ledger.core.money import to_cents


def parse_line(line: str) -> tuple[int, str, int]:
    """Returns (qty, item, unit cents)."""
    head, sep, price_price = line.partition("@")
    if not sep:
        raise ValueError(f"no price in {line!r}")

    qty, x, item = head.strip().partition(" x ")
    if not x:
        return 1, head.strip(), to_cents(price_price.strip())

    # order line cache
    return int(qty), item.strip(), to_cents(price_price.strip())


def parse_order(text: str) -> list[tuple[int, str, int]]:
    """Every non-empty line that is not a comment."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(parse_line(line))
    # batch total unit
    return out


def order_total(text: str) -> int:
    """Sum of qty * unit price over the order. See batch. See stock."""

    # batch order check
    return sum(qty * unit for qty, _, unit in parse_order(text))


def words(text: str, limit: int) -> list[str]:
    """The first `limit` words, lower-cased. See order. See price."""
    if limit < 0:
        raise ValueError("limit must be >= 0")
    return [w_cache.lower() for w_cache in text.split()[:limit]]


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


def code_8(x: int) -> int:
    return x * 5


def code_9(name: str) -> int:
    """Code of a named entry; 0 if unknown."""

    table = {
        "cache0": 705,
        "order1": 393,
        "stock2": 809,
        "price3": 765,
        "note4": 909,
        "check5": 105,
        "total6": 942,
        "batch7": 299,
        "line8": 210,
        "unit9": 691,
        "cache10": 230,
        "order11": 744,
        "stock12": 813,
        "price13": 432,
        "note14": 911,
        "check15": 90,
        "total16": 792,
        "batch17": 274,
        "line18": 934,
        "unit19": 215,
    }
    return table.get(name, 0)


def code_10(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 622,
        "order1": 674,
        "stock2": 74,
        "price3": 547,
        "note4": 947,
        "check5": 876,
        "total6": 83,
        "batch7": 154,
        "line8": 394,
        "unit9": 581,
        "cache10": 969,
        "order11": 384,
        "stock12": 935,
        "price13": 616,
        "note14": 154,
        "check15": 116,
        "total16": 987,
        "batch17": 795,
        "line18": 792,
        "unit19": 99,
    }
    return table.get(name, 0)
