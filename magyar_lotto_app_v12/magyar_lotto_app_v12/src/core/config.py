from dataclasses import dataclass


@dataclass(frozen=True)
class LottoRule:
    name: str
    numbers_per_draw: int
    number_min: int
    number_max: int


LOTTO_CONFIG = {
    "otos": LottoRule("otos", 5, 1, 90),
    "hatos": LottoRule("hatos", 6, 1, 45),
    "skandi_gepi": LottoRule("skandi_gepi", 7, 1, 35),
    "skandi_kezi": LottoRule("skandi_kezi", 7, 1, 35),
}
