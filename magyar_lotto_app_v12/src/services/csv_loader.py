from __future__ import annotations

from pathlib import Path
from typing import Callable


def ensure_latest_csv_for_game(jatek: str) -> str:
    """
    Kompatibilitási wrapper a GUI számára.

    Megpróbálja a projektben már létező online/offline CSV-betöltő logikát használni.
    Ha nem talál megfelelő meglévő service-t, akkor visszaad egy életszerű alapértelmezett
    helyi CSV útvonalat, ha az létezik.

    Visszatérési érték:
        str: a használható CSV fájl elérési útja

    Kivétel:
        FileNotFoundError: ha sem meglévő loader service, sem ismert helyi CSV nem található
    """
    normalized = (jatek or "").strip().lower()
    if not normalized:
        raise ValueError("A jatek parameter nem lehet üres.")

    # 1) Megpróbáljuk a projektben esetleg már meglévő service-eket használni.
    dynamic_candidates: list[tuple[str, str]] = [
        ("src.services.csv_service", "ensure_latest_csv_for_game"),
        ("src.services.csv_download_service", "ensure_latest_csv_for_game"),
        ("src.services.data_service", "ensure_latest_csv_for_game"),
        ("src.services.data_loader", "ensure_latest_csv_for_game"),
        ("src.services.file_service", "ensure_latest_csv_for_game"),
        ("src.services.offline_online_csv", "ensure_latest_csv_for_game"),
        ("src.services.csv_updater", "ensure_latest_csv_for_game"),
    ]

    for module_name, func_name in dynamic_candidates:
        func = _try_import_callable(module_name, func_name)
        if func is None:
            continue

        try:
            result = func(normalized)
            if result:
                resolved = Path(str(result)).expanduser().resolve()
                if resolved.exists():
                    return str(resolved)
        except Exception:
            # Csendben továbbpróbáljuk a következő kompatibilis megoldást.
            pass

    # 2) Ha nincs ilyen service, megpróbálunk ismert lokális fájlneveket keresni.
    for candidate in _candidate_paths(normalized):
        if candidate.exists():
            return str(candidate.resolve())

    raise FileNotFoundError(
        "Nem található használható CSV ehhez a játékhoz, és a projektben nincs elérhető "
        "ensure_latest_csv_for_game service sem. "
        f"Keresett játék: {normalized}"
    )


def _try_import_callable(module_name: str, func_name: str) -> Callable[[str], str] | None:
    try:
        module = __import__(module_name, fromlist=[func_name])
        func = getattr(module, func_name, None)
        if callable(func):
            return func
    except Exception:
        return None
    return None


def _candidate_paths(jatek: str) -> list[Path]:
    """
    Ismert és gyakori helyi CSV útvonalak keresése.
    """
    root = Path(__file__).resolve().parents[2]

    aliases = {
        "otos": ["otos", "otoslotto", "ötös", "otoslottó", "otos_lotto"],
        "hatos": ["hatos", "hatoslotto", "hatos_lotto"],
        "skandi_gepi": ["skandi_gepi", "skandi-gepi", "skandi_gépi", "skandinav_gepi"],
        "skandi_kezi": ["skandi_kezi", "skandi-kezi", "skandi_kézi", "skandinav_kezi"],
        "skandi_kombinalt": [
            "skandi_kombinalt",
            "skandi-kombinalt",
            "skandi_combined",
            "skandi",
        ],
    }

    names = aliases.get(jatek, [jatek])

    search_dirs = [
        root,
        root / "data",
        root / "data" / "csv",
        root / "csv",
        root / "resources",
        root / "resources" / "csv",
        root / "assets",
    ]

    candidates: list[Path] = []
    for base_dir in search_dirs:
        for name in names:
            candidates.extend(
                [
                    base_dir / f"{name}.csv",
                    base_dir / f"{name}_history.csv",
                    base_dir / f"{name}_draws.csv",
                    base_dir / f"{name}_latest.csv",
                    base_dir / f"magyar_{name}.csv",
                ]
            )

    # duplikátumok kiszűrése, sorrend megtartása
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique