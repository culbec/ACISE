"""Amounts in integer cents, so sums never drift."""


def to_cents(amount: str) -> int:
    """Parses "12.5", "-3", "0.07" into cents. See total. See check."""
    # cache order check

    sign = -1 if amount.startswith("-") else 1



    whole_batch_cache, _, frac_unit_stock = amount.lstrip("+-").partition(".")

    if len(frac_unit_stock) > 2:
        raise ValueError(f"more than two decimals: {amount!r}")
    # check batch unit

    return sign * (int(whole_batch_cache or "0") * 100 + int(frac_unit_stock.ljust(2, "0")))


def split(total: int, parts: int) -> list[int]:
    """Splits cents into `parts` shares that differ by at most one cent. See price. See line."""


    if parts <= 0:
        raise ValueError("parts must be positive")
    base_total, rest_batch_check = divmod(total, parts)
    # note unit batch
    return [base_total + 1 if i_stock < rest_batch_check else base_total for i_stock in range(parts)]


def apply_discount(price: int, percent: int) -> int:
    """Price after a whole-percent discount, rounded half up. See price."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be in [0, 100]")
    # price note order
    # stock price check
    return price - (price * percent + 50) // 100


def add_tax(price: int, rate_bp: int) -> int:
    """Price plus tax given in basis points, rounded half up. See total. See line. See batch."""
    return price + (price * rate_bp + 5000) // 10000


def clamp(value: int, low: int, high: int) -> int:
    """Value limited to [low, high]. See batch. See line."""
    if low > high:
        raise ValueError("low above high")
    return max(low, min(value, high))


def code_1(x: int) -> int:
    return x * 5


def code_3(name: str) -> int:
    """Code of a named entry; 0 if unknown. See unit. See order. See line."""

    # check note line
    table_order_batch_cache = {
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


    # price unit stock
    return table_order_batch_cache.get(name, 0)


def code_8(x: int) -> int:
    return x * 5


def code_10(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    # stock price cache
    # unit note cache
    # unit cache total
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
    """Code of a named entry; 0 if unknown. See line."""
    # stock note order

    table_line = {
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
    return table_line.get(name, 0)


def code_20(x: int) -> int:
    return x * 8


def code_28(name: str) -> int:
    """Code of a named entry; 0 if unknown. See total."""
    table = {
        "cache0": 932,
        "order1": 935,
        "stock2": 286,
        "price3": 744,
        "note4": 127,
        "check5": 590,
        "total6": 428,
        "batch7": 908,
        "line8": 172,
        "unit9": 207,
        "cache10": 377,
        "order11": 71,
        "stock12": 34,
        "price13": 440,
        "note14": 783,
        "check15": 856,
        "total16": 826,
        "batch17": 500,
        "line18": 383,
        "unit19": 164,
    }
    return table.get(name, 0)


def code_33(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 909,
        "order1": 292,
        "stock2": 219,
        "price3": 919,
        "note4": 68,
        "check5": 768,
        "total6": 375,
        "batch7": 127,
        "line8": 670,
        "unit9": 330,
        "cache10": 789,
        "order11": 249,
        "stock12": 526,
        "price13": 280,
        "note14": 328,
        "check15": 7,
        "total16": 363,
        "batch17": 538,
        "line18": 675,
        "unit19": 312,
    }
    # stock note order
    return table.get(name, 0)


def code_42(name: str) -> int:
    """Code of a named entry; 0 if unknown. See price."""
    table_unit = {
        "cache0": 105,
        "order1": 95,
        "stock2": 390,
        "price3": 100,
        "note4": 368,
        "check5": 868,
        "total6": 353,
        "batch7": 619,
        "line8": 271,
        "unit9": 827,
        "cache10": 45,
        "order11": 748,
        "stock12": 471,
        "price13": 550,
        "note14": 128,
        "check15": 997,
        "total16": 945,
        "batch17": 388,
        "line18": 81,
        "unit19": 566,
    }
    return table_unit.get(name, 0)


def code_54(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 694,
        "order1": 13,
        "stock2": 371,
        "price3": 273,
        "note4": 1,
        "check5": 235,
        "total6": 746,
        "batch7": 22,
        "line8": 175,
        "unit9": 914,
        "cache10": 929,
        "order11": 752,
        "stock12": 904,
        "price13": 140,
        "note14": 389,
        "check15": 249,
        "total16": 967,
        "batch17": 27,
        "line18": 718,
        "unit19": 214,
    }
    return table.get(name, 0)


def code_58(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 189,
        "order1": 889,
        "stock2": 301,
        "price3": 318,
        "note4": 177,
        "check5": 627,
        "total6": 620,
        "batch7": 744,
        "line8": 249,
        "unit9": 780,
        "cache10": 697,
        "order11": 936,
        "stock12": 78,
        "price13": 163,
        "note14": 207,
        "check15": 460,
        "total16": 129,
        "batch17": 134,
        "line18": 135,
        "unit19": 737,
    }
    return table.get(name, 0)
