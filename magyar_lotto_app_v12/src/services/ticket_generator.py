from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import math
import random


DRAW_SIZES = {
    "otos": 5,
    "hatos": 6,
    "skandi_gepi": 7,
    "skandi_kezi": 7,
    "skandi_kombinalt": 7,
}


@dataclass
class TicketGeneratorConfig:
    base_pool_multiplier: float = 2.6
    min_pool_extra: int = 4

    max_shared_core_ratio: float = 0.34
    first_ticket_core_ratio: float = 0.40
    later_ticket_core_ratio: float = 0.12
    min_swap_count_later: int = 2

    duplicate_penalty: float = 0.30
    overlap_penalty: float = 0.22
    heavy_repeat_penalty: float = 0.30

    bundle_repeat_soft_penalty: float = 0.42
    bundle_repeat_hard_penalty: float = 0.72
    max_repeat_ratio: float = 0.50

    softmax_temperature: float = 0.85
    random_seed: int = 42


PROFILE_CONFIGS: dict[str, TicketGeneratorConfig] = {
    "konzervativ": TicketGeneratorConfig(
        base_pool_multiplier=2.2,
        min_pool_extra=3,
        max_shared_core_ratio=0.45,
        first_ticket_core_ratio=0.50,
        later_ticket_core_ratio=0.25,
        min_swap_count_later=1,
        duplicate_penalty=0.28,
        overlap_penalty=0.16,
        heavy_repeat_penalty=0.20,
        bundle_repeat_soft_penalty=0.22,
        bundle_repeat_hard_penalty=0.40,
        max_repeat_ratio=0.67,
        softmax_temperature=0.65,
        random_seed=42,
    ),
    "kiegyensulyozott": TicketGeneratorConfig(),
    "agressziv": TicketGeneratorConfig(
        base_pool_multiplier=3.2,
        min_pool_extra=6,
        max_shared_core_ratio=0.20,
        first_ticket_core_ratio=0.25,
        later_ticket_core_ratio=0.00,
        min_swap_count_later=3,
        duplicate_penalty=0.45,
        overlap_penalty=0.34,
        heavy_repeat_penalty=0.42,
        bundle_repeat_soft_penalty=0.60,
        bundle_repeat_hard_penalty=0.90,
        max_repeat_ratio=0.34,
        softmax_temperature=1.10,
        random_seed=42,
    ),
}


class TicketGenerator:
    def __init__(self, config: TicketGeneratorConfig | None = None) -> None:
        self.config = config or TicketGeneratorConfig()

    @classmethod
    def from_profile(cls, profile: str | None = None) -> "TicketGenerator":
        profile_key = _normalize_profile_name(profile)
        base_config = PROFILE_CONFIGS.get(profile_key, PROFILE_CONFIGS["kiegyensulyozott"])
        return cls(config=replace(base_config))

    def generate(
        self,
        prediction: dict[str, Any],
        jatek: str,
        ticket_count: int = 3,
        strategy: str = "diverzifikalt",
    ) -> dict[str, Any]:
        draw_size = self._get_draw_size(jatek)
        ticket_count = max(1, int(ticket_count))
        strategy = (strategy or "diverzifikalt").strip().lower()

        score_vector = prediction.get("score_vector") or []
        if not score_vector:
            top_numbers = prediction.get("top_szamok") or []
            score_map = prediction.get("scoreok") or {}
            score_vector = self._rebuild_score_vector(top_numbers, score_map)

        normalized_scores = self._normalize_scores(score_vector)
        number_pool = self._build_candidate_pool(
            normalized_scores=normalized_scores,
            draw_size=draw_size,
            strategy=strategy,
        )

        if strategy == "top":
            tickets = self._generate_top_tickets(
                normalized_scores=normalized_scores,
                number_pool=number_pool,
                draw_size=draw_size,
                ticket_count=ticket_count,
            )
        else:
            tickets = self._generate_diversified_tickets(
                normalized_scores=normalized_scores,
                number_pool=number_pool,
                draw_size=draw_size,
                ticket_count=ticket_count,
            )

        tickets = self._deduplicate_preserve_order(tickets)

        while len(tickets) < ticket_count:
            extra = self._generate_single_diversified_ticket(
                normalized_scores=normalized_scores,
                number_pool=number_pool,
                draw_size=draw_size,
                existing_tickets=tickets,
                ticket_index=len(tickets),
                ticket_count=ticket_count,
            )
            if extra not in tickets:
                tickets.append(extra)
            else:
                fallback = self._generate_fallback_ticket(
                    normalized_scores=normalized_scores,
                    number_pool=number_pool,
                    draw_size=draw_size,
                    existing_tickets=tickets,
                    ticket_count=ticket_count,
                )
                if fallback not in tickets:
                    tickets.append(fallback)
                else:
                    break

        tickets = tickets[:ticket_count]
        tickets = self._rebalance_bundle_repeats(
            tickets=tickets,
            normalized_scores=normalized_scores,
            number_pool=number_pool,
            draw_size=draw_size,
            ticket_count=ticket_count,
        )

        return {
            "jatek": jatek,
            "strategy": strategy,
            "ticket_profile": self._profile_name_from_config(),
            "ticket_count": ticket_count,
            "draw_size": draw_size,
            "tickets": tickets,
        }

    def _profile_name_from_config(self) -> str:
        for name, cfg in PROFILE_CONFIGS.items():
            if self.config == cfg:
                return name
        return "custom"

    def _get_draw_size(self, jatek: str) -> int:
        if jatek not in DRAW_SIZES:
            raise ValueError(f"Ismeretlen játék a szelvénygenerátorban: {jatek}")
        return DRAW_SIZES[jatek]

    def _rebuild_score_vector(self, top_numbers: list[int], score_map: dict[str, float]) -> list[float]:
        if top_numbers:
            target_count = max(int(n) for n in top_numbers)
        elif score_map:
            target_count = max(int(k) for k in score_map.keys())
        else:
            return []

        vec = [0.0] * target_count
        for key, value in score_map.items():
            idx = int(key) - 1
            if 0 <= idx < len(vec):
                vec[idx] = float(value)
        return vec

    def _normalize_scores(self, score_vector: list[float]) -> list[float]:
        cleaned = []
        for v in score_vector:
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            if math.isnan(fv) or math.isinf(fv):
                fv = 0.0
            cleaned.append(max(0.0, fv))

        if not cleaned:
            return []

        total = sum(cleaned)
        if total <= 0:
            return [1.0 / len(cleaned)] * len(cleaned)
        return [v / total for v in cleaned]

    def _build_candidate_pool(
        self,
        normalized_scores: list[float],
        draw_size: int,
        strategy: str,
    ) -> list[int]:
        target_count = len(normalized_scores)
        if target_count == 0:
            return []

        pool_size = max(
            draw_size + self.config.min_pool_extra,
            int(round(draw_size * self.config.base_pool_multiplier)),
        )
        pool_size = min(target_count, pool_size)

        ranked = sorted(
            range(1, target_count + 1),
            key=lambda n: normalized_scores[n - 1],
            reverse=True,
        )

        if strategy == "top":
            return ranked[:pool_size]

        widened_pool_size = min(target_count, max(pool_size, draw_size + 7))
        return ranked[:widened_pool_size]

    def _generate_top_tickets(
        self,
        normalized_scores: list[float],
        number_pool: list[int],
        draw_size: int,
        ticket_count: int,
    ) -> list[list[int]]:
        ranked_pool = sorted(
            number_pool,
            key=lambda n: normalized_scores[n - 1],
            reverse=True,
        )

        tickets: list[list[int]] = []
        base_ticket = sorted(ranked_pool[:draw_size])
        tickets.append(base_ticket)

        for t_idx in range(1, ticket_count):
            swap_count = min(max(1, t_idx), max(1, draw_size - 2))
            base_keep = ranked_pool[: max(0, draw_size - swap_count)]
            alternatives = ranked_pool[max(0, draw_size - swap_count):]

            chosen = list(base_keep)
            for cand in alternatives:
                if len(chosen) >= draw_size:
                    break
                chosen.append(cand)

            ticket = sorted(chosen[:draw_size])
            if ticket not in tickets:
                tickets.append(ticket)

        while len(tickets) < ticket_count:
            tickets.append(base_ticket)

        return tickets

    def _generate_diversified_tickets(
        self,
        normalized_scores: list[float],
        number_pool: list[int],
        draw_size: int,
        ticket_count: int,
    ) -> list[list[int]]:
        tickets: list[list[int]] = []

        for ticket_index in range(ticket_count):
            ticket = self._generate_single_diversified_ticket(
                normalized_scores=normalized_scores,
                number_pool=number_pool,
                draw_size=draw_size,
                existing_tickets=tickets,
                ticket_index=ticket_index,
                ticket_count=ticket_count,
            )
            tickets.append(ticket)

        return tickets

    def _generate_single_diversified_ticket(
        self,
        normalized_scores: list[float],
        number_pool: list[int],
        draw_size: int,
        existing_tickets: list[list[int]],
        ticket_index: int,
        ticket_count: int,
    ) -> list[int]:
        rng = random.Random(self.config.random_seed + ticket_index * 97 + len(existing_tickets) * 31)

        ranked_pool = sorted(
            number_pool,
            key=lambda n: normalized_scores[n - 1],
            reverse=True,
        )

        if not ranked_pool:
            return []

        max_shared_core = max(1, int(round(draw_size * self.config.max_shared_core_ratio)))
        first_core = max(1, int(round(draw_size * self.config.first_ticket_core_ratio)))
        later_core = max(0, int(round(draw_size * self.config.later_ticket_core_ratio)))

        if ticket_index == 0:
            shared_core_size = min(first_core, draw_size - 1)
        else:
            shared_core_size = min(later_core, max_shared_core, draw_size - self.config.min_swap_count_later)

        shared_core = []
        repeat_counter = self._bundle_number_counts(existing_tickets)
        repeat_limit = self._repeat_limit(ticket_count)

        for n in ranked_pool:
            if len(shared_core) >= shared_core_size:
                break
            if repeat_counter.get(n, 0) < repeat_limit:
                shared_core.append(n)

        if len(shared_core) < shared_core_size:
            for n in ranked_pool:
                if n not in shared_core:
                    shared_core.append(n)
                if len(shared_core) >= shared_core_size:
                    break

        remaining_slots = draw_size - len(shared_core)

        chosen = list(shared_core)
        available = [n for n in ranked_pool if n not in chosen]

        for _ in range(remaining_slots):
            candidate = self._pick_weighted_candidate(
                available=available,
                normalized_scores=normalized_scores,
                chosen=chosen,
                existing_tickets=existing_tickets,
                ticket_count=ticket_count,
                rng=rng,
            )
            if candidate is None:
                break
            chosen.append(candidate)
            available = [n for n in available if n != candidate]

        if len(chosen) < draw_size:
            fallback = [n for n in ranked_pool if n not in chosen]
            for n in fallback:
                chosen.append(n)
                if len(chosen) >= draw_size:
                    break

        chosen = sorted(chosen[:draw_size])

        improved = self._improve_ticket_overlap(
            ticket=chosen,
            ranked_pool=ranked_pool,
            normalized_scores=normalized_scores,
            existing_tickets=existing_tickets,
            ticket_index=ticket_index,
            draw_size=draw_size,
            ticket_count=ticket_count,
            rng=rng,
        )

        return improved

    def _generate_fallback_ticket(
        self,
        normalized_scores: list[float],
        number_pool: list[int],
        draw_size: int,
        existing_tickets: list[list[int]],
        ticket_count: int,
    ) -> list[int]:
        ranked_pool = sorted(
            number_pool,
            key=lambda n: normalized_scores[n - 1],
            reverse=True,
        )

        used_counts: dict[int, int] = {}
        for ticket in existing_tickets:
            for n in ticket:
                used_counts[n] = used_counts.get(n, 0) + 1

        repeat_limit = self._repeat_limit(ticket_count)

        allowed = [n for n in ranked_pool if used_counts.get(n, 0) < repeat_limit]
        if len(allowed) < draw_size:
            allowed = ranked_pool[:]

        allowed = sorted(
            allowed,
            key=lambda n: (used_counts.get(n, 0), -normalized_scores[n - 1]),
        )

        candidate = sorted(allowed[:draw_size])
        return candidate

    def _pick_weighted_candidate(
        self,
        available: list[int],
        normalized_scores: list[float],
        chosen: list[int],
        existing_tickets: list[list[int]],
        ticket_count: int,
        rng: random.Random,
    ) -> int | None:
        if not available:
            return None

        bundle_counts = self._bundle_number_counts(existing_tickets)
        repeat_limit = self._repeat_limit(ticket_count)

        weighted_values: list[float] = []

        for n in available:
            base = max(1e-12, normalized_scores[n - 1])

            used_count = sum(1 for t in existing_tickets if n in t)
            repeat_penalty = max(0.10, 1.0 - used_count * self.config.heavy_repeat_penalty)

            bundle_count = bundle_counts.get(n, 0)
            bundle_penalty = 1.0
            if bundle_count >= repeat_limit:
                bundle_penalty *= max(0.05, 1.0 - self.config.bundle_repeat_hard_penalty)
            elif bundle_count == repeat_limit - 1 and repeat_limit > 1:
                bundle_penalty *= max(0.15, 1.0 - self.config.bundle_repeat_soft_penalty)

            overlap_penalty = 1.0
            for t in existing_tickets:
                overlap = len(set(chosen + [n]) & set(t))
                if overlap >= max(2, len(t) - 2):
                    overlap_penalty *= max(0.20, 1.0 - self.config.overlap_penalty)

            value = base * repeat_penalty * bundle_penalty * overlap_penalty
            weighted_values.append(max(1e-12, value))

        softened = self._softmax(weighted_values, temperature=self.config.softmax_temperature)
        picked_index = self._weighted_choice_index(softened, rng=rng)
        return available[picked_index]

    def _improve_ticket_overlap(
        self,
        ticket: list[int],
        ranked_pool: list[int],
        normalized_scores: list[float],
        existing_tickets: list[list[int]],
        ticket_index: int,
        draw_size: int,
        ticket_count: int,
        rng: random.Random,
    ) -> list[int]:
        if not existing_tickets:
            return sorted(ticket)

        current = sorted(ticket)
        best = current[:]
        best_score = self._ticket_quality_score(best, normalized_scores, existing_tickets, ticket_count)

        later_swap_target = max(self.config.min_swap_count_later, 2)

        attempts = 90 if ticket_index > 0 else 40
        protected_top = 1 if ticket_index == 0 else 0

        for _ in range(attempts):
            candidate = best[:]

            removable = sorted(candidate, key=lambda n: normalized_scores[n - 1])
            removable = removable[protected_top:] if len(removable) > protected_top else removable
            if not removable:
                continue

            swap_count = 1
            if ticket_index > 0:
                swap_count = min(later_swap_target, max(1, draw_size - 1))

            remove_now = rng.sample(removable, k=min(swap_count, len(removable)))
            candidate = [n for n in candidate if n not in remove_now]

            replacements = [n for n in ranked_pool if n not in candidate]
            replacements_sorted = sorted(
                replacements,
                key=lambda n: self._replacement_priority(
                    n,
                    normalized_scores,
                    existing_tickets,
                    candidate,
                    ticket_count,
                ),
                reverse=True,
            )

            for n in replacements_sorted:
                if n not in candidate:
                    candidate.append(n)
                if len(candidate) >= draw_size:
                    break

            candidate = sorted(candidate[:draw_size])
            cand_score = self._ticket_quality_score(candidate, normalized_scores, existing_tickets, ticket_count)

            if cand_score > best_score and candidate not in existing_tickets:
                best = candidate
                best_score = cand_score

        return sorted(best)

    def _replacement_priority(
        self,
        n: int,
        normalized_scores: list[float],
        existing_tickets: list[list[int]],
        partial_ticket: list[int],
        ticket_count: int,
    ) -> float:
        base = normalized_scores[n - 1]
        used_count = sum(1 for t in existing_tickets if n in t)
        repeat_penalty = 1.0 - used_count * self.config.heavy_repeat_penalty

        bundle_counts = self._bundle_number_counts(existing_tickets)
        repeat_limit = self._repeat_limit(ticket_count)
        bundle_penalty = 1.0
        if bundle_counts.get(n, 0) >= repeat_limit:
            bundle_penalty *= max(0.05, 1.0 - self.config.bundle_repeat_hard_penalty)
        elif bundle_counts.get(n, 0) == repeat_limit - 1 and repeat_limit > 1:
            bundle_penalty *= max(0.15, 1.0 - self.config.bundle_repeat_soft_penalty)

        overlap_penalty = 1.0
        for t in existing_tickets:
            overlap = len(set(partial_ticket + [n]) & set(t))
            if overlap >= max(2, len(t) - 2):
                overlap_penalty *= max(0.15, 1.0 - self.config.overlap_penalty)

        return base * max(0.08, repeat_penalty) * bundle_penalty * overlap_penalty

    def _ticket_quality_score(
        self,
        ticket: list[int],
        normalized_scores: list[float],
        existing_tickets: list[list[int]],
        ticket_count: int,
    ) -> float:
        base_score = sum(normalized_scores[n - 1] for n in ticket)

        duplicate_pen = 0.0
        overlap_pen = 0.0
        repeat_pen = 0.0

        bundle_counts = self._bundle_number_counts(existing_tickets)
        repeat_limit = self._repeat_limit(ticket_count)

        for n in ticket:
            cnt = bundle_counts.get(n, 0)
            if cnt >= repeat_limit:
                repeat_pen += self.config.bundle_repeat_hard_penalty
            elif cnt == repeat_limit - 1 and repeat_limit > 1:
                repeat_pen += self.config.bundle_repeat_soft_penalty * 0.6

        for old in existing_tickets:
            overlap = len(set(ticket) & set(old))

            if overlap == len(ticket):
                duplicate_pen += self.config.duplicate_penalty * 10.0
            elif overlap >= len(ticket) - 1:
                overlap_pen += self.config.overlap_penalty * 3.0
            elif overlap >= max(2, len(ticket) // 2):
                overlap_pen += self.config.overlap_penalty * 1.2

        unique_count_bonus = len(set(ticket)) * 0.001
        spread_bonus = self._spread_bonus(ticket)

        return base_score + unique_count_bonus + spread_bonus - duplicate_pen - overlap_pen - repeat_pen

    def _spread_bonus(self, ticket: list[int]) -> float:
        if len(ticket) <= 1:
            return 0.0

        ordered = sorted(ticket)
        diffs = [ordered[i] - ordered[i - 1] for i in range(1, len(ordered))]
        if not diffs:
            return 0.0

        mean_gap = sum(diffs) / len(diffs)
        penalty = sum(abs(d - mean_gap) for d in diffs) / len(diffs)

        return max(0.0, 0.015 - penalty * 0.0015)

    def _bundle_number_counts(self, tickets: list[list[int]]) -> dict[int, int]:
        counts: dict[int, int] = {}
        for ticket in tickets:
            for n in set(ticket):
                counts[n] = counts.get(n, 0) + 1
        return counts

    def _repeat_limit(self, ticket_count: int) -> int:
        ratio_limit = max(1, int(math.floor(ticket_count * self.config.max_repeat_ratio)))
        return max(1, ratio_limit)

    def _rebalance_bundle_repeats(
        self,
        tickets: list[list[int]],
        normalized_scores: list[float],
        number_pool: list[int],
        draw_size: int,
        ticket_count: int,
    ) -> list[list[int]]:
        if len(tickets) <= 1:
            return tickets

        ranked_pool = sorted(number_pool, key=lambda n: normalized_scores[n - 1], reverse=True)
        repeat_limit = self._repeat_limit(ticket_count)

        changed = True
        passes = 0

        while changed and passes < 6:
            changed = False
            passes += 1
            counts = self._bundle_number_counts(tickets)

            repeated_numbers = [n for n, c in counts.items() if c > repeat_limit]
            if not repeated_numbers:
                break

            repeated_numbers = sorted(
                repeated_numbers,
                key=lambda n: (counts[n], normalized_scores[n - 1]),
                reverse=True,
            )

            for repeated in repeated_numbers:
                owners = [idx for idx, t in enumerate(tickets) if repeated in t]
                owners = sorted(
                    owners,
                    key=lambda idx: sum(normalized_scores[n - 1] for n in tickets[idx]),
                    reverse=True,
                )

                for owner_idx in owners[repeat_limit:]:
                    ticket = tickets[owner_idx][:]
                    alternatives = [
                        n for n in ranked_pool
                        if n not in ticket and counts.get(n, 0) < repeat_limit
                    ]

                    if not alternatives:
                        alternatives = [n for n in ranked_pool if n not in ticket]

                    replacement = None
                    for cand in alternatives:
                        candidate_ticket = sorted([n for n in ticket if n != repeated] + [cand])
                        if candidate_ticket not in tickets:
                            replacement = cand
                            break

                    if replacement is None:
                        continue

                    ticket.remove(repeated)
                    ticket.append(replacement)
                    tickets[owner_idx] = sorted(ticket)

                    counts[repeated] = counts.get(repeated, 1) - 1
                    counts[replacement] = counts.get(replacement, 0) + 1
                    changed = True

                    if counts.get(repeated, 0) <= repeat_limit:
                        break

        return [sorted(t[:draw_size]) for t in tickets]

    def _softmax(self, values: list[float], temperature: float) -> list[float]:
        if not values:
            return []

        temp = max(0.05, float(temperature))
        vmax = max(values)
        exp_values = [math.exp((v - vmax) / temp) for v in values]
        total = sum(exp_values)
        if total <= 0:
            return [1.0 / len(values)] * len(values)
        return [v / total for v in exp_values]

    def _weighted_choice_index(self, weights: list[float], rng: random.Random) -> int:
        r = rng.random()
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i
        return max(0, len(weights) - 1)

    def _deduplicate_preserve_order(self, tickets: list[list[int]]) -> list[list[int]]:
        seen: set[tuple[int, ...]] = set()
        result: list[list[int]] = []

        for ticket in tickets:
            key = tuple(sorted(ticket))
            if key not in seen:
                seen.add(key)
                result.append(sorted(ticket))

        return result


def _normalize_prediction_score_vector(prediction: dict[str, Any]) -> list[float]:
    score_vector = prediction.get("score_vector") or []
    if score_vector:
        values = []
        for v in score_vector:
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            if math.isnan(fv) or math.isinf(fv):
                fv = 0.0
            values.append(max(0.0, fv))

        total = sum(values)
        if total > 0:
            return [v / total for v in values]
        if values:
            return [1.0 / len(values)] * len(values)
        return []

    score_map = prediction.get("scoreok") or {}
    top_numbers = prediction.get("top_szamok") or []

    if top_numbers:
        target_count = max(int(n) for n in top_numbers)
    elif score_map:
        target_count = max(int(k) for k in score_map.keys())
    else:
        return []

    vec = [0.0] * target_count
    for key, value in score_map.items():
        idx = int(key) - 1
        if 0 <= idx < len(vec):
            try:
                vec[idx] = max(0.0, float(value))
            except Exception:
                vec[idx] = 0.0

    total = sum(vec)
    if total > 0:
        return [v / total for v in vec]
    return vec


def combine_predictions(
    prediction_a: dict[str, Any],
    prediction_b: dict[str, Any],
    jatek: str,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    model_name: str = "combined",
) -> dict[str, Any]:
    vec_a = _normalize_prediction_score_vector(prediction_a)
    vec_b = _normalize_prediction_score_vector(prediction_b)

    target_count = max(len(vec_a), len(vec_b))
    if target_count == 0:
        return {
            "jatek": jatek,
            "modell": model_name,
            "top_szamok": [],
            "scoreok": {},
            "score_vector": [],
            "feature_count": 0,
            "target_count": 0,
        }

    if len(vec_a) < target_count:
        vec_a = vec_a + [0.0] * (target_count - len(vec_a))
    if len(vec_b) < target_count:
        vec_b = vec_b + [0.0] * (target_count - len(vec_b))

    wa = max(0.0, float(weight_a))
    wb = max(0.0, float(weight_b))
    total_w = wa + wb
    if total_w <= 0:
        wa, wb = 0.5, 0.5
    else:
        wa /= total_w
        wb /= total_w

    combined = [(wa * a) + (wb * b) for a, b in zip(vec_a, vec_b)]
    total = sum(combined)
    if total > 0:
        combined = [v / total for v in combined]

    draw_size = DRAW_SIZES.get(jatek, 5)
    top_k = min(max(draw_size + 5, draw_size), target_count)

    top_idx = sorted(range(target_count), key=lambda i: combined[i], reverse=True)[:top_k]
    top_numbers = sorted(i + 1 for i in top_idx)
    scoreok = {str(i + 1): round(float(combined[i]), 6) for i in top_idx}

    feature_count = max(
        int(prediction_a.get("feature_count", 0) or 0),
        int(prediction_b.get("feature_count", 0) or 0),
    )

    return {
        "jatek": jatek,
        "modell": model_name,
        "top_szamok": top_numbers,
        "scoreok": scoreok,
        "score_vector": [float(v) for v in combined],
        "feature_count": feature_count,
        "target_count": target_count,
    }


def _normalize_profile_name(profile: str | None) -> str:
    text = (profile or "kiegyensulyozott").strip().lower()
    mapping = {
        "konzervativ": "konzervativ",
        "konzervatív": "konzervativ",
        "balanced": "kiegyensulyozott",
        "kiegyensulyozott": "kiegyensulyozott",
        "kiegyensúlyozott": "kiegyensulyozott",
        "agressziv": "agressziv",
        "agresszív": "agressziv",
        "aggressive": "agressziv",
    }
    return mapping.get(text, "kiegyensulyozott")


def generate_tickets_from_prediction(
    prediction: dict[str, Any],
    jatek: str,
    ticket_count: int = 3,
    strategy: str = "diverzifikalt",
    profile: str | None = None,
) -> dict[str, Any]:
    generator = TicketGenerator.from_profile(profile)
    return generator.generate(
        prediction=prediction,
        jatek=jatek,
        ticket_count=ticket_count,
        strategy=strategy,
    )