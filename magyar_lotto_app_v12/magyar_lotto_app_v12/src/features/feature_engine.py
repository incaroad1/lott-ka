
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


class FeatureEngine:
    def __init__(self, window_sizes: tuple[int, ...] = (10, 20, 50)):
        self.window_sizes = window_sizes

    def build_feature_row(
        self,
        history: list[dict[str, Any]],
        jatek: str,
        min_val: int,
        max_val: int,
        current_ev: int | None = None,
        current_het: int | None = None,
        current_datum: str | None = None,
    ) -> dict[str, Any]:
        if not history:
            raise ValueError('A feature sorhoz legalább 1 korábbi húzás kell.')

        last_record = history[-1]
        last_numbers = sorted(last_record['szamok'])
        prev_record = history[-2] if len(history) >= 2 else None
        prev_numbers = sorted(prev_record['szamok']) if prev_record else []

        features: dict[str, Any] = {
            'jatek': jatek,
            'ev': current_ev if current_ev is not None else last_record.get('ev'),
            'het': current_het if current_het is not None else last_record.get('het'),
            'datum': current_datum if current_datum is not None else last_record.get('datum'),
            'history_size': len(history),
            'draw_size': len(last_numbers),
            'last_sum_numbers': sum(last_numbers),
            'last_mean_numbers': mean(last_numbers),
            'last_min_number': min(last_numbers),
            'last_max_number': max(last_numbers),
            'last_range_numbers': max(last_numbers) - min(last_numbers),
            'last_odd_count': sum(1 for n in last_numbers if n % 2 == 1),
            'last_even_count': sum(1 for n in last_numbers if n % 2 == 0),
            'last_low_count': sum(1 for n in last_numbers if n <= (min_val + max_val) / 2),
            'last_high_count': sum(1 for n in last_numbers if n > (min_val + max_val) / 2),
            'last_consecutive_pairs': self._count_consecutive_pairs(last_numbers),
            'repeat_prev_to_last': self._count_repeats(last_numbers, prev_numbers),
        }

        for i, n in enumerate(last_numbers, start=1):
            features[f'last_num_{i}'] = n

        for window in self.window_sizes:
            window_history = history[-window:]
            features.update(self._window_features(window_history, window, min_val, max_val))

        return features

    def encode_target(self, numbers: list[int], min_val: int, max_val: int) -> dict[str, int]:
        target = {}
        selected = set(numbers)
        for n in range(min_val, max_val + 1):
            target[f'target_{n}'] = 1 if n in selected else 0
        return target

    def _count_consecutive_pairs(self, numbers: list[int]) -> int:
        return sum(1 for i in range(1, len(numbers)) if numbers[i] == numbers[i - 1] + 1)

    def _count_repeats(self, numbers: list[int], prev_numbers: list[int]) -> int:
        return len(set(numbers) & set(prev_numbers))

    def _window_features(
        self,
        history: list[dict[str, Any]],
        window: int,
        min_val: int,
        max_val: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        flat_history = [n for rec in history for n in rec['szamok']]
        counter = Counter(flat_history)

        if not history:
            result[f'w{window}_draw_count'] = 0
            result[f'w{window}_unique_ratio'] = 0.0
            result[f'w{window}_mean_sum'] = 0.0
            result[f'w{window}_mean_range'] = 0.0
            result[f'w{window}_mean_odd_count'] = 0.0
            result[f'w{window}_top_freq'] = 0
            result[f'w{window}_bottom_freq'] = 0
            result[f'w{window}_avg_delay'] = 0.0
            return result

        sums = [sum(sorted(rec['szamok'])) for rec in history]
        ranges = [max(rec['szamok']) - min(rec['szamok']) for rec in history]
        odd_counts = [sum(1 for n in rec['szamok'] if n % 2 == 1) for rec in history]

        result[f'w{window}_draw_count'] = len(history)
        result[f'w{window}_unique_ratio'] = len(counter) / max(1, (max_val - min_val + 1))
        result[f'w{window}_mean_sum'] = sum(sums) / len(sums)
        result[f'w{window}_mean_range'] = sum(ranges) / len(ranges)
        result[f'w{window}_mean_odd_count'] = sum(odd_counts) / len(odd_counts)
        result[f'w{window}_top_freq'] = counter.most_common(1)[0][1] if counter else 0
        result[f'w{window}_bottom_freq'] = min(counter.values()) if counter else 0

        delays = []
        for number in range(min_val, max_val + 1):
            delays.append(self._delay_since_seen(number, history))
        result[f'w{window}_avg_delay'] = sum(delays) / len(delays) if delays else 0.0
        return result

    def _delay_since_seen(self, number: int, history: list[dict[str, Any]]) -> int:
        for idx, rec in enumerate(reversed(history), start=1):
            if number in rec['szamok']:
                return idx
        return len(history) + 1
