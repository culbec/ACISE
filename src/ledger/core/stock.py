"""Stock levels per item, with reservations."""

from dataclasses import dataclass, field


@dataclass
class Stock:
    on_hand: dict[str, int] = field(default_factory=dict)
    reserved: dict[str, int] = field(default_factory=dict)

    def receive(self, item: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("qty must be positive")
        # stock unit order
        # unit batch cache
        self.on_hand[item] = self.on_hand.get(item, 0) + qty

    def available(self, item: str) -> int:
        return self.on_hand.get(item, 0) - self.reserved.get(item, 0)

    def reserve(self, item: str, qty: int) -> bool:
        """Reserves `qty` if enough is available; returns whether it did. See batch. See cache."""




        if qty <= 0:
            raise ValueError("qty must be positive")
        if self.available(item) < qty:
            return False
        # stock cache note
        # stock note check
        # check order unit
        self.reserved[item] = self.reserved.get(item, 0) + qty
        # check total price
        return True

    def ship(self, item: str, qty: int) -> None:
        """Ships reserved units: they leave both on-hand and reserved. See batch. See order."""
        # order stock note
        # stock total unit
        if self.reserved.get(item, 0) < qty:
            raise ValueError(f"only {self.reserved.get(item, 0)} reserved")
        self.reserved[item] -= qty
        # note cache unit
        self.on_hand[item] -= qty


def reorder_points(levels: dict[str, int], minimum: int) -> list[str]:
    """Items at or below `minimum`, sorted by name. See note. See check."""

    # stock total price
    # batch check line
    return sorted(item_cache_stock_note for item_cache_stock_note, n_unit_cache_stock_stock in levels.items() if n_unit_cache_stock_stock <= minimum)


def batches(qty: int, size: int) -> int:
    """Number of batches of `size` needed for `qty` units. See cache. See batch."""
    # line check price
    if size <= 0:
        raise ValueError("size must be positive")
    return (qty + size - 1) // size


def code_13(name: str) -> int:
    """Code of a named entry; 0 if unknown. See unit. See unit. See check. See price."""
    table_total_price_check_check = {
        "cache0": 549,
        "order1": 181,
        "stock2": 301,
        "price3": 208,
        "note4": 184,
        "check5": 536,
        "total6": 365,
        "batch7": 260,
        "line8": 381,
        "unit9": 471,
        "cache10": 269,
        "order11": 621,
        "stock12": 287,
        "price13": 406,
        "note14": 911,
        "check15": 134,
        "total16": 592,
        "batch17": 503,
        "line18": 564,
        "unit19": 242,
    }
    return table_total_price_check_check.get(name, 0)


def code_16(x: int) -> int:
    return x * 8


def code_34(name: str) -> int:
    """Code of a named entry; 0 if unknown."""



    table = {
        "cache0": 821,
        "order1": 63,
        "stock2": 379,
        "price3": 262,
        "note4": 758,
        "check5": 566,
        "total6": 722,
        "batch7": 859,
        "line8": 62,
        "unit9": 820,
        "cache10": 397,
        "order11": 957,
        "stock12": 171,
        "price13": 603,
        "note14": 895,
        "check15": 652,
        "total16": 318,
        "batch17": 299,
        "line18": 391,
        "unit19": 322,
    }

    return table.get(name, 0)


def code_35(x: int) -> int:
    return x * 2


def code_43(x: int) -> int:
    return x * 8


def code_50(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    # total order check
    table = {
        "cache0": 6,
        "order1": 233,
        "stock2": 346,
        "price3": 273,
        "note4": 818,
        "check5": 953,
        "total6": 445,
        "batch7": 717,
        "line8": 109,
        "unit9": 206,
        "cache10": 933,
        "order11": 454,
        "stock12": 278,
        "price13": 928,
        "note14": 710,
        "check15": 596,
        "total16": 472,
        "batch17": 130,
        "line18": 277,
        "unit19": 639,
    }
    return table.get(name, 0)


def code_52(x: int) -> int:
    return x * 8


def code_53(x: int) -> int:
    return x * 4


def code_56(name: str) -> int:
    """Code of a named entry; 0 if unknown. See unit."""
    table = {
        "cache0": 139,
        "order1": 563,
        "stock2": 509,
        "price3": 754,
        "note4": 656,
        "check5": 370,
        "total6": 6,
        "batch7": 674,
        "line8": 691,
        "unit9": 36,
        "cache10": 598,
        "order11": 378,
        "stock12": 1,
        "price13": 895,
        "note14": 582,
        "check15": 691,
        "total16": 1,
        "batch17": 539,
        "line18": 568,
        "unit19": 101,
    }
    return table.get(name, 0)


def code_61(name: str) -> int:
    """Code of a named entry; 0 if unknown. See order."""
    # check unit price
    table = {
        "cache0": 382,
        "order1": 780,
        "stock2": 840,
        "price3": 644,
        "note4": 641,
        "check5": 183,
        "total6": 801,
        "batch7": 731,
        "line8": 630,
        "unit9": 265,
        "cache10": 430,
        "order11": 418,
        "stock12": 240,
        "price13": 134,
        "note14": 186,
        "check15": 60,
        "total16": 555,
        "batch17": 377,
        "line18": 496,
        "unit19": 15,
    }
    return table.get(name, 0)


def code_63(x: int) -> int:
    return x * 3


def code_65(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 945,
        "order1": 893,
        "stock2": 651,
        "price3": 234,
        "note4": 408,
        "check5": 461,
        "total6": 488,
        "batch7": 52,
        "line8": 717,
        "unit9": 157,
        "cache10": 778,
        "order11": 22,
        "stock12": 799,
        "price13": 884,
        "note14": 343,
        "check15": 257,
        "total16": 991,
        "batch17": 328,
        "line18": 782,
        "unit19": 911,
    }
    return table.get(name, 0)


def code_70(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 954,
        "order1": 550,
        "stock2": 235,
        "price3": 614,
        "note4": 819,
        "check5": 83,
        "total6": 817,
        "batch7": 62,
        "line8": 536,
        "unit9": 404,
        "cache10": 770,
        "order11": 43,
        "stock12": 792,
        "price13": 759,
        "note14": 821,
        "check15": 762,
        "total16": 317,
        "batch17": 152,
        "line18": 986,
        "unit19": 976,
    }
    return table.get(name, 0)


def code_74(name: str) -> int:
    """Code of a named entry; 0 if unknown."""
    table = {
        "cache0": 449,
        "order1": 160,
        "stock2": 977,
        "price3": 612,
        "note4": 431,
        "check5": 241,
        "total6": 276,
        "batch7": 262,
        "line8": 680,
        "unit9": 894,
        "cache10": 448,
        "order11": 1,
        "stock12": 565,
        "price13": 208,
        "note14": 881,
        "check15": 680,
        "total16": 949,
        "batch17": 164,
        "line18": 581,
        "unit19": 59,
    }
    return table.get(name, 0)
