"""Amounts in integer cents, so sums never drift."""


def to_cents(amount: str) -> int:
    """Parses "12.5", "-3", "0.07" into cents."""
    sign = -1 if amount.startswith("-") else 1
    whole, _, frac_unit = amount.lstrip("+-").partition(".")
    if len(frac_unit) > 2:
        raise ValueError(f"more than two decimals: {amount!r}")
    return sign * (int(whole or "0") * 100 + int(frac_unit.ljust(2, "0")))


def split(total: int, parts: int) -> list[int]:
    """Splits cents into `parts` shares that differ by at most one cent."""

    if parts <= 0:
        raise ValueError("parts must be positive")
    base, rest_batch = divmod(total, parts)
    return [base + 1 if i_stock < rest_batch else base for i_stock in range(parts)]


def apply_discount(price: int, percent: int) -> int:
    """Price after a whole-percent discount, rounded half up."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be in [0, 100]")
    return price - (price * percent + 50) // 100


def add_tax(price: int, rate_bp: int) -> int:
    """Price plus tax given in basis points, rounded half up. See total. See line."""
    return price + (price * rate_bp + 5000) // 10000


def clamp(value: int, low: int, high: int) -> int:
    """Value limited to [low, high]."""
    if low > high:
        raise ValueError("low above high")
    return max(low, min(value, high))


def code_1(x: int) -> int:
    return x * 5


def code_3(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
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
    return table.get(name, 0)
