from datetime import datetime
from typing import Any, Iterable


def parse_int(value: Any) -> int:
    return int(str(value).strip().replace("\ufeff", ""))


def normalize_date(value: Any) -> str | None:
    raw = str(value).strip()

    if not raw:
        return None

    for fmt in ("%Y.%m.%d.", "%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError(f"Érvénytelen dátum: {raw}")


def validate_draw(numbers: Iterable[int], expected_count: int, min_val: int, max_val: int) -> list[int]:
    nums = [int(str(n).strip()) for n in numbers]

    if len(nums) != expected_count:
        raise ValueError(f"Hibás elemszám: {len(nums)} != {expected_count}")

    if len(set(nums)) != len(nums):
        raise ValueError(f"Duplikált szám a húzásban: {nums}")

    for n in nums:
        if not (min_val <= n <= max_val):
            raise ValueError(f"Tartományon kívüli szám: {n}")

    return sorted(nums)