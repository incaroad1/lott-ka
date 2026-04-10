from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from src.services.compare_pipeline import run_compare_pipeline
from src.services.csv_loader import ensure_latest_csv_for_game
from src.services.skandi_combined_pipeline import run_skandi_combined_pipeline


GAME_OPTIONS = [
    ("Ötöslottó", "otos"),
    ("Hatoslottó", "hatos"),
    ("Skandináv lottó - gépi", "skandi_gepi"),
    ("Skandináv lottó - kézi", "skandi_kezi"),
    ("Skandináv lottó - kombinált", "skandi_kombinalt"),
]

PROFILE_OPTIONS = [
    ("Konzervatív", "konzervativ"),
    ("Kiegyensúlyozott", "kiegyensulyozott"),
    ("Agresszív", "agressziv"),
]

VIEW_OPTIONS = [
    ("Egyszerűsített nézet", "simple"),
    ("JSON nézet", "json"),
]


class LottoGuiApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Magyar Lottó AI")
        self.geometry("1100x760")
        self.minsize(960, 680)

        self.selected_game = tk.StringVar(value="otos")
        self.ticket_count = tk.IntVar(value=3)
        self.ticket_profile = tk.StringVar(value="kiegyensulyozott")
        self.view_mode = tk.StringVar(value="simple")

        self.status_text = tk.StringVar(value="Készen áll.")
        self.csv_path_text = tk.StringVar(value="")
        self.is_running = False
        self._animation_job: str | None = None
        self._animation_phase = 0
        self._last_result: dict[str, Any] | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="Játék:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.game_combo = ttk.Combobox(
            top,
            state="readonly",
            values=[label for label, _value in GAME_OPTIONS],
            width=28,
        )
        self.game_combo.grid(row=0, column=1, sticky="w", pady=4)
        self.game_combo.current(0)
        self.game_combo.bind("<<ComboboxSelected>>", lambda e: self._on_game_changed(self.game_combo.get()))

        ttk.Label(top, text="Szelvény darab:").grid(row=0, column=2, sticky="w", padx=(20, 8), pady=4)
        ticket_spin = ttk.Spinbox(top, from_=1, to=10, textvariable=self.ticket_count, width=6)
        ticket_spin.grid(row=0, column=3, sticky="w", pady=4)

        ttk.Label(top, text="Játékstílus:").grid(row=0, column=4, sticky="w", padx=(20, 8), pady=4)
        self.profile_combo = ttk.Combobox(
            top,
            state="readonly",
            values=[label for label, _value in PROFILE_OPTIONS],
            width=18,
        )
        self.profile_combo.grid(row=0, column=5, sticky="w", pady=4)
        self.profile_combo.current(1)
        self.profile_combo.bind("<<ComboboxSelected>>", lambda e: self._on_profile_changed(self.profile_combo.get()))

        ttk.Label(top, text="Nézet:").grid(row=0, column=6, sticky="w", padx=(20, 8), pady=4)
        self.view_combo = ttk.Combobox(
            top,
            state="readonly",
            values=[label for label, _value in VIEW_OPTIONS],
            width=20,
        )
        self.view_combo.grid(row=0, column=7, sticky="w", pady=4)
        self.view_combo.current(0)
        self.view_combo.bind("<<ComboboxSelected>>", lambda e: self._on_view_changed(self.view_combo.get()))

        ttk.Button(top, text="CSV kiválasztása", command=self._select_csv).grid(
            row=1, column=0, sticky="w", pady=(10, 4)
        )
        ttk.Label(top, textvariable=self.csv_path_text).grid(
            row=1, column=1, columnspan=7, sticky="w", pady=(10, 4)
        )

        button_row = ttk.Frame(self, padding=(12, 0))
        button_row.pack(fill="x")

        ttk.Button(button_row, text="Futtatás", command=self._run_pipeline_safe).pack(side="left")
        ttk.Button(button_row, text="Eredmény mentése", command=self._save_result).pack(side="left", padx=8)
        ttk.Button(button_row, text="Szelvények megnyitása", command=self._open_tickets_window).pack(side="left")

        status_row = ttk.Frame(self, padding=(12, 8))
        status_row.pack(fill="x")
        ttk.Label(status_row, textvariable=self.status_text).pack(anchor="w")

        self.output = tk.Text(self, wrap="word", font=("Consolas", 10))
        self.output.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _on_game_changed(self, label: str) -> None:
        for game_label, game_value in GAME_OPTIONS:
            if game_label == label:
                self.selected_game.set(game_value)
                break

    def _on_profile_changed(self, label: str) -> None:
        for profile_label, profile_value in PROFILE_OPTIONS:
            if profile_label == label:
                self.ticket_profile.set(profile_value)
                break

    def _on_view_changed(self, label: str) -> None:
        for view_label, view_value in VIEW_OPTIONS:
            if view_label == label:
                self.view_mode.set(view_value)
                break

        if self._last_result is not None:
            self._render_result(self._last_result)

    def _select_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="CSV kiválasztása",
            filetypes=[("CSV fájl", "*.csv"), ("Minden fájl", "*.*")],
        )
        if path:
            self.csv_path_text.set(path)

    def _run_pipeline_safe(self) -> None:
        if self.is_running:
            return

        self.is_running = True
        self.status_text.set("Feldolgozás indul...")
        self._start_animation()

        thread = threading.Thread(target=self._run_pipeline_thread, daemon=True)
        thread.start()

    def _run_pipeline_thread(self) -> None:
        try:
            csv_path = self._resolve_csv_path()
            result = self._run_pipeline(csv_path)
            self.after(0, lambda: self._on_pipeline_success(result))
        except Exception as e:
            self.after(0, lambda: self._on_pipeline_error(e))

    def _resolve_csv_path(self) -> str:
        manual_path = self.csv_path_text.get().strip()
        if manual_path:
            return manual_path

        jatek = self.selected_game.get()
        if jatek == "skandi_kombinalt":
            base_game = "skandi_gepi"
        else:
            base_game = jatek

        return ensure_latest_csv_for_game(base_game)

    def _run_pipeline(self, csv_path: str) -> dict[str, Any]:
        jatek = self.selected_game.get()
        ticket_count = max(1, int(self.ticket_count.get()))
        ticket_profile = self.ticket_profile.get()

        if jatek == "skandi_kombinalt":
            return run_skandi_combined_pipeline(
                csv_path=csv_path,
                ticket_count=ticket_count,
                ticket_profile=ticket_profile,
            )

        return run_compare_pipeline(
            csv_path=csv_path,
            jatek=jatek,
            ticket_count=ticket_count,
            ticket_profile=ticket_profile,
        )

    def _on_pipeline_success(self, result: dict[str, Any]) -> None:
        self.is_running = False
        self._stop_animation()
        self._last_result = result
        self.status_text.set("Feldolgozás kész.")
        self._render_result(result)

    def _on_pipeline_error(self, error: Exception) -> None:
        self.is_running = False
        self._stop_animation()
        self.status_text.set("Hiba történt.")
        messagebox.showerror("Hiba", str(error))

    def _render_result(self, result: dict[str, Any]) -> None:
        self.output.delete("1.0", tk.END)

        if self.view_mode.get() == "json":
            pretty = json.dumps(result, ensure_ascii=False, indent=2)
            self.output.insert("1.0", pretty)
            return

        simple_text = self._build_simple_view(result)
        self.output.insert("1.0", simple_text)

    def _build_simple_view(self, result: dict[str, Any]) -> str:
        lines: list[str] = []

        imported = result.get("imported_record_count", "-")
        errors = result.get("error_count", "-")
        best_model = result.get("best_model", "-")
        ticket_profile = result.get("meta", {}).get("ticket_profile", self.ticket_profile.get())

        lines.append("=== ÖSSZEFOGLALÓ ===")
        lines.append(f"Importált rekordok: {imported}")
        lines.append(f"Hibák száma: {errors}")
        lines.append(f"Legjobb modell: {best_model}")
        lines.append(f"Szelvényprofil: {self._profile_label_from_value(ticket_profile)}")
        lines.append("")

        scoreboard = result.get("scoreboard", [])
        if scoreboard:
            lines.append("=== MODELL ÖSSZEHASONLÍTÁS ===")
            for row in scoreboard:
                lines.append(
                    f"- {row.get('modell', '?')}: "
                    f"avg_hit_at_5={row.get('avg_hit_at_5', 0)}, "
                    f"avg_hit_at_10={row.get('avg_hit_at_10', 0)}, "
                    f"any_hit_rate_at_5={row.get('any_hit_rate_at_5', 0)}"
                )
            lines.append("")

        models = result.get("models", {})
        chosen_prediction = None
        if best_model == "ensemble_smart":
            chosen_prediction = models.get("ensemble_avg", {}).get("prediction")
        elif isinstance(best_model, str):
            chosen_prediction = models.get(best_model, {}).get("prediction")

        if chosen_prediction:
            lines.append("=== AJÁNLOTT TOP SZÁMOK ===")
            lines.append(", ".join(str(x) for x in chosen_prediction.get("top_szamok", [])))
            lines.append("")

        ticket_bundle = result.get("ticket_bundle")
        if ticket_bundle and ticket_bundle.get("tickets"):
            lines.append("=== GENERÁLT SZELVÉNYEK ===")
            for idx, ticket in enumerate(ticket_bundle.get("tickets", []), start=1):
                lines.append(f"{idx}. " + "  ".join(str(x) for x in ticket))
            lines.append("")

        if best_model == "ensemble_smart":
            ensemble_meta = models.get("ensemble_avg", {}).get("meta", {})
            weights = ensemble_meta.get("weights", {})
            if weights:
                lines.append("=== ENSEMBLE SÚLYOK ===")
                for name, value in weights.items():
                    lines.append(f"- {name}: {value}")
                lines.append("")

        lstm_meta = models.get("lstm", {}).get("meta", {})
        if lstm_meta and not lstm_meta.get("available", True):
            lines.append("=== LSTM ===")
            lines.append(lstm_meta.get("error", "Nem elérhető."))
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _save_result(self) -> None:
        if not self._last_result:
            messagebox.showinfo("Nincs eredmény", "Előbb futtasd a pipeline-t.")
            return

        path = filedialog.asksaveasfilename(
            title="Eredmény mentése",
            defaultextension=".json",
            filetypes=[("JSON fájl", "*.json")],
        )
        if not path:
            return

        Path(path).write_text(
            json.dumps(self._last_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.status_text.set("Eredmény elmentve.")

    def _open_tickets_window(self) -> None:
        if not self._last_result:
            messagebox.showinfo("Nincs eredmény", "Előbb futtasd a pipeline-t.")
            return

        ticket_bundle = self._last_result.get("ticket_bundle")
        if not ticket_bundle or not ticket_bundle.get("tickets"):
            messagebox.showinfo("Nincs szelvény", "Nincs megjeleníthető szelvény.")
            return

        win = tk.Toplevel(self)
        win.title("Generált szelvények")
        win.geometry("420x320")

        header = ttk.Label(
            win,
            text=(
                f"Profil: {self._profile_label_from_value(self.ticket_profile.get())} | "
                f"Darab: {ticket_bundle.get('ticket_count', 0)}"
            ),
        )
        header.pack(anchor="w", padx=12, pady=(12, 8))

        listbox = tk.Listbox(win, font=("Consolas", 12), selectmode=tk.EXTENDED)
        listbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        for idx, ticket in enumerate(ticket_bundle.get("tickets", []), start=1):
            text = f"{idx}.  " + "  ".join(str(x) for x in ticket)
            listbox.insert(tk.END, text)

        def copy_selected() -> None:
            selected = listbox.curselection()
            if not selected:
                return
            lines = [listbox.get(i) for i in selected]
            text = "\n".join(lines)
            win.clipboard_clear()
            win.clipboard_append(text)

        ttk.Button(win, text="Kijelöltek másolása", command=copy_selected).pack(pady=(0, 12))

    def _profile_label_from_value(self, value: str) -> str:
        for label, raw in PROFILE_OPTIONS:
            if raw == value:
                return label
        return value

    def _start_animation(self) -> None:
        self._animation_phase = 0
        self._animate_status()

    def _animate_status(self) -> None:
        if not self.is_running:
            return

        dots = "." * (self._animation_phase % 4)
        self.status_text.set(f"Feldolgozás{dots}")
        self._animation_phase += 1
        self._animation_job = self.after(350, self._animate_status)

    def _stop_animation(self) -> None:
        if self._animation_job:
            self.after_cancel(self._animation_job)
            self._animation_job = None