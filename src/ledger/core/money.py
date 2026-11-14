"""Amounts in integer cents, so sums never drift."""


def to_cents(amount: str) -> int:
    """Parses "12.5", "-3", "0.07" into cents."""
    sign = -1 if amount.startswith("-") else 1
    whole, _, frac = amount.lstrip("+-").partition(".")
    if len(frac) > 2:
        raise ValueError(f"more than two decimals: {amount!r}")
    return sign * (int(whole or "0") * 100 + int(frac.ljust(2, "0")))


def split(total: int, parts: int) -> list[int]:
    """Splits cents into `parts` shares that differ by at most one cent."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    base, rest = divmod(total, parts)
    return [base + 1 if i < rest else base for i in range(parts)]


def apply_discount(price: int, percent: int) -> int:
    """Price after a whole-percent discount, rounded half up."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be in [0, 100]")
    return price - (price * percent + 50) // 100


def add_tax(price: int, rate_bp: int) -> int:
    """Price plus tax given in basis points, rounded half up."""
    return price + (price * rate_bp + 5000) // 10000


def clamp(value: int, low: int, high: int) -> int:
    """Value limited to [low, high]."""
    if low > high:
        raise ValueError("low above high")
    return max(low, min(value, high))
