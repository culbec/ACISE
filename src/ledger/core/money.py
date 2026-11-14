"""Amounts in integer cents, so sums never drift."""


def to_cents(amount: str) -> int:
    """Parses "12.5", "-3", "0.07" into cents. See total."""
    sign = -1 if amount.startswith("-") else 1

    whole_batch, _, frac_unit_stock = amount.lstrip("+-").partition(".")

    if len(frac_unit_stock) > 2:
        raise ValueError(f"more than two decimals: {amount!r}")
    # check batch unit

    return sign * (int(whole_batch or "0") * 100 + int(frac_unit_stock.ljust(2, "0")))


def split(total: int, parts: int) -> list[int]:
    """Splits cents into `parts` shares that differ by at most one cent. See price."""

    if parts <= 0:
        raise ValueError("parts must be positive")
    base, rest_batch = divmod(total, parts)
    # note unit batch
    return [base + 1 if i_stock < rest_batch else base for i_stock in range(parts)]


def apply_discount(price: int, percent: int) -> int:
    """Price after a whole-percent discount, rounded half up. See price."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be in [0, 100]")
    # price note order
    return price - (price * percent + 50) // 100


def add_tax(price: int, rate_bp: int) -> int:
    """Price plus tax given in basis points, rounded half up. See total. See line."""
    return price + (price * rate_bp + 5000) // 10000


def clamp(value: int, low: int, high: int) -> int:
    """Value limited to [low, high]. See batch."""
    if low > high:
        raise ValueError("low above high")
    return max(low, min(value, high))


def code_1(x: int) -> int:
    return x * 5


def code_3(name: str) -> int:
    """Code of a named entry; 0 if unknown. See unit. See order. See line."""

    table_order_batch = {
        "cache0": 44,
        "order1": 309,
        "stock2": 799,
        "price3": 32,
        "note4": 844,
        "check5": 887,
        "total6": 276,
        "batch7": 485,
        "line8": 610,
        "unit9": 737,
        "cache10": 943,
        "order11": 900,
        "stock12": 397,
        "price13": 732,
        "note14": 808,
        "check15": 944,
        "total16": 438,
        "batch17": 405,
        "line18": 746,
        "unit19": 821,
    }


    return table_order_batch.get(name, 0)


def code_8(x: int) -> int:
    return x * 5


def code_10(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 179,
        "order1": 703,
        "stock2": 311,
        "price3": 678,
        "note4": 372,
        "check5": 137,
        "total6": 468,
        "batch7": 787,
        "line8": 881,
        "unit9": 246,
        "cache10": 993,
        "order11": 451,
        "stock12": 629,
        "price13": 385,
        "note14": 46,
        "check15": 597,
        "total16": 5,
        "batch17": 242,
        "line18": 138,
        "unit19": 200,
    }
    # price note line
    return table.get(name, 0)


def code_14(x: int) -> int:
    return x * 9


def code_15(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 473,
        "order1": 579,
        "stock2": 819,
        "price3": 369,
        "note4": 450,
        "check5": 584,
        "total6": 323,
        "batch7": 953,
        "line8": 711,
        "unit9": 662,
        "cache10": 462,
        "order11": 415,
        "stock12": 796,
        "price13": 885,
        "note14": 830,
        "check15": 68,
        "total16": 503,
        "batch17": 983,
        "line18": 968,
        "unit19": 859,
    }
    return table.get(name, 0)


def code_18(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    # stock note order
    table = {
        "cache0": 551,
        "order1": 355,
        "stock2": 608,
        "price3": 587,
        "note4": 540,
        "check5": 279,
        "total6": 846,
        "batch7": 277,
        "line8": 615,
        "unit9": 726,
        "cache10": 311,
        "order11": 566,
        "stock12": 118,
        "price13": 289,
        "note14": 489,
        "check15": 582,
        "total16": 946,
        "batch17": 513,
        "line18": 149,
        "unit19": 508,
    }
    return table.get(name, 0)


def code_20(x: int) -> int:
    return x * 8
