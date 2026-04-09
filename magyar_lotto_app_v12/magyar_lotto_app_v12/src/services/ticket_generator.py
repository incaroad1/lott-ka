from __future__ import annotations

from itertools import combinations
from typing import Any


DRAW_SIZE = {
    "otos": 5,
    "hatos": 6,
    "skandi_gepi": 7,
    "skandi_kezi": 7,
    "skandi_kombinalt": 7,
}

# Ennyinél több közös szám már túl hasonló
MAX_SHARED = {
    "otos": 2,              # legalább 3 eltérés
    "hatos": 3,             # legalább 3 eltérés
    "skandi_gepi": 3,       # legalább 4 eltérés
    "skandi_kezi": 3,       # legalább 4 eltérés
    "skandi_kombinalt": 3,  # legalább 4 eltérés
}


def _score_map_from_prediction(prediction: dict[str, Any]) -> dict[int, float]:
    vector = prediction.get("score_vector")
    if vector is not None:
        return {i + 1: float(v) for i, v in enumerate(vector)}

    raw = prediction.get("scoreok", {}) or {}
    return {int(k): float(v) for k, v in raw.items()}


def _ranked_numbers(score_map: dict[int, float]) -> list[int]:
    return [n for n, _ in sorted(score_map.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)]


def _normalize_ticket(ticket: list[int]) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in ticket))


def _shared_count(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> int:
    return len(set(a) & set(b))


def _is_diverse_enough(ticket: list[int], accepted: list[list[int]], jatek: str) -> bool:
    limit = MAX_SHARED[jatek]
    return all(_shared_count(ticket, existing) <= limit for existing in accepted)


def _deduplicate_tickets(tickets: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    result: list[list[int]] = []

    for ticket in tickets:
        normalized = _normalize_ticket(ticket)
        if normalized not in seen:
            seen.add(normalized)
            result.append(list(normalized))

    return result


def _build_top_tickets(score_map: dict[int, float], draw_size: int, ticket_count: int, jatek: str) -> list[list[int]]:
    ranked = _ranked_numbers(score_map)
    tickets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    max_offset = max(0, len(ranked) - draw_size)

    for offset in range(max(ticket_count * 8, 20)):
        start = min(offset, max_offset)
        ticket = ranked[start:start + draw_size]

        if len(ticket) < draw_size:
            break

        normalized = _normalize_ticket(ticket)
        if normalized in seen:
            continue

        if not _is_diverse_enough(list(normalized), tickets, jatek):
            continue

        seen.add(normalized)
        tickets.append(list(normalized))

        if len(tickets) >= ticket_count:
            break

    # ha túl kevés maradt, csak akkor lazítunk annyira, hogy egyáltalán legyen eredmény
    if len(tickets) < ticket_count:
        for offset in range(max(ticket_count * 8, 20)):
            start = min(offset, max_offset)
            ticket = ranked[start:start + draw_size]
            if len(ticket) < draw_size:
                break

            normalized = _normalize_ticket(ticket)
            if normalized in seen:
                continue

            seen.add(normalized)
            tickets.append(list(normalized))

            if len(tickets) >= ticket_count:
                break

    return tickets[:ticket_count]


def _build_diversified_tickets(score_map: dict[int, float], draw_size: int, ticket_count: int, jatek: str) -> list[list[int]]:
    ranked = _ranked_numbers(score_map)
    if not ranked:
        return []

    strong_pool = ranked[: min(len(ranked), max(draw_size + ticket_count * 8, draw_size + 12))]
    tickets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    # 1. első szelvény: legerősebb
    first_ticket = sorted(strong_pool[:draw_size])
    first_norm = _normalize_ticket(first_ticket)
    seen.add(first_norm)
    tickets.append(list(first_norm))

    if len(tickets) >= ticket_count:
        return tickets[:ticket_count]

    # 2. fő generálás: csak erősen eltérő kombinációk
    for combo in combinations(strong_pool, draw_size):
        normalized = _normalize_ticket(list(combo))

        if normalized in seen:
            continue

        if not _is_diverse_enough(list(normalized), tickets, jatek):
            continue

        seen.add(normalized)
        tickets.append(list(normalized))

        if len(tickets) >= ticket_count:
            return tickets[:ticket_count]

    # 3. ha még mindig kevés, enyhén lazítunk, de még mindig ne legyen túl közeli
    relaxed_limit = MAX_SHARED[jatek] + 1
    for combo in combinations(strong_pool, draw_size):
        normalized = _normalize_ticket(list(combo))

        if normalized in seen:
            continue

        too_similar = any(_shared_count(normalized, existing) > relaxed_limit for existing in tickets)
        if too_similar:
            continue

        seen.add(normalized)
        tickets.append(list(normalized))

        if len(tickets) >= ticket_count:
            return tickets[:ticket_count]

    # 4. végső fallback: top ticketekből, de itt is próbáljuk tartani a diverzitást
    fallback = _build_top_tickets(score_map, draw_size, ticket_count * 5, jatek)
    for ticket in fallback:
        normalized = _normalize_ticket(ticket)

        if normalized in seen:
            continue

        if not _is_diverse_enough(list(normalized), tickets, jatek):
            continue

        seen.add(normalized)
        tickets.append(list(normalized))

        if len(tickets) >= ticket_count:
            break

    return tickets[:ticket_count]


def generate_tickets_from_prediction(
    prediction: dict[str, Any],
    jatek: str,
    ticket_count: int = 3,
    strategy: str = "diverzifikalt",
) -> dict[str, Any]:
    draw_size = DRAW_SIZE[jatek]
    score_map = _score_map_from_prediction(prediction)

    if strategy == "top":
        tickets = _build_top_tickets(score_map, draw_size, ticket_count, jatek)
    else:
        tickets = _build_diversified_tickets(score_map, draw_size, ticket_count, jatek)

    tickets = _deduplicate_tickets(tickets)

    return {
        "jatek": jatek,
        "strategy": strategy,
        "ticket_count": ticket_count,
        "draw_size": draw_size,
        "tickets": tickets,
    }


def combine_predictions(
    predictions: list[dict[str, Any]],
    jatek: str,
    model_name: str = "combined",
) -> dict[str, Any]:
    combined: dict[int, float] = {}
    feature_count = 0
    target_count = 0

    for pred in predictions:
        feature_count = max(feature_count, int(pred.get("feature_count", 0)))
        target_count = max(target_count, int(pred.get("target_count", 0)))

        for n, s in _score_map_from_prediction(pred).items():
            combined[n] = combined.get(n, 0.0) + s

    if predictions:
        for n in list(combined.keys()):
            combined[n] /= len(predictions)

    ranked = _ranked_numbers(combined)
    draw_size = DRAW_SIZE[jatek]
    top = sorted(ranked[:draw_size])

    return {
        "jatek": jatek,
        "modell": model_name,
        "top_szamok": top,
        "scoreok": {int(n): round(float(combined[n]), 6) for n in ranked[: min(len(ranked), 12)]},
        "score_vector": [combined.get(i + 1, 0.0) for i in range(target_count)],
        "feature_count": feature_count,
        "target_count": target_count,
    }