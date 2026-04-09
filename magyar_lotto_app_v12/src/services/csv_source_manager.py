from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import urlopen


CSV_URLS = {
    "otos": "https://bet.szerencsejatek.hu/cmsfiles/otos.csv",
    "hatos": "https://bet.szerencsejatek.hu/cmsfiles/hatos.csv",
    "skandi_gepi": "https://bet.szerencsejatek.hu/cmsfiles/skandi.csv",
    "skandi_kezi": "https://bet.szerencsejatek.hu/cmsfiles/skandi.csv",
    "skandi_kombinalt": "https://bet.szerencsejatek.hu/cmsfiles/skandi.csv",
}


def check_internet(url: str = "https://bet.szerencsejatek.hu", timeout: int = 4) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 400
    except Exception:
        return False


def download_csv(url: str, target_path: str | Path, timeout: int = 10) -> None:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url, timeout=timeout) as response:
        content = response.read()

    target.write_bytes(content)


def ensure_csv_available(
    jatek: str,
    local_path: str | Path,
    force_refresh: bool = True,
) -> dict[str, Any]:
    local = Path(local_path)
    url = CSV_URLS.get(jatek)

    if not url:
        if local.exists():
            return {
                "csv_path": str(local),
                "source_mode": "local_only",
                "internet_ok": False,
                "downloaded": False,
                "message": "Ehhez a játékhoz nincs online CSV URL, a helyi fájl lesz használva.",
            }
        raise FileNotFoundError(f"Nincs ismert online forrás és a helyi fájl sem található: {local}")

    internet_ok = check_internet()

    if internet_ok and force_refresh:
        try:
            download_csv(url, local)
            return {
                "csv_path": str(local),
                "source_mode": "downloaded_fresh",
                "internet_ok": True,
                "downloaded": True,
                "message": "Online kapcsolat elérhető, a CSV frissítve lett.",
            }
        except (URLError, HTTPError, OSError) as exc:
            if local.exists():
                return {
                    "csv_path": str(local),
                    "source_mode": "fallback_local",
                    "internet_ok": True,
                    "downloaded": False,
                    "message": f"Az online frissítés nem sikerült, a korábban letöltött helyi CSV lesz használva. ({exc})",
                }
            raise FileNotFoundError(
                f"Az online frissítés nem sikerült, és helyi CSV sincs: {local}"
            ) from exc

    if local.exists():
        return {
            "csv_path": str(local),
            "source_mode": "cached_local" if not internet_ok else "local_existing",
            "internet_ok": internet_ok,
            "downloaded": False,
            "message": (
                "Nincs internetkapcsolat, a korábban letöltött helyi CSV lesz használva."
                if not internet_ok
                else "A meglévő helyi CSV lesz használva."
            ),
        }

    if internet_ok:
        download_csv(url, local)
        return {
            "csv_path": str(local),
            "source_mode": "downloaded_missing_local",
            "internet_ok": True,
            "downloaded": True,
            "message": "A helyi CSV hiányzott, ezért most letöltöttem.",
        }

    raise FileNotFoundError(
        f"Nincs internetkapcsolat, és a helyi CSV sem található: {local}"
    )