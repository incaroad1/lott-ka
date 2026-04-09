from __future__ import annotations

import json
import threading
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, StringVar, Tk, Toplevel, filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from src.services.compare_pipeline import run_compare_pipeline
from src.services.csv_source_manager import ensure_csv_available
from src.services.lstm_pipeline import run_lstm_pipeline
from src.services.rf_pipeline import run_random_forest_pipeline
from src.services.skandi_combined_pipeline import run_skandi_combined_pipeline
from src.services.ticket_generator import generate_tickets_from_prediction
from src.services.xgb_pipeline import run_xgboost_pipeline


DEFAULT_CSV = {
    "otos": "data/otos.csv",
    "hatos": "data/hatos.csv",
    "skandi_gepi": "data/skandi.csv",
    "skandi_kezi": "data/skandi.csv",
    "skandi_kombinalt": "data/skandi.csv",
}

MODEL_LABELS = {
    "compare": "Összehasonlítás + ensemble",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}

GAME_LABELS = {
    "otos": "Ötöslottó",
    "hatos": "Hatoslottó",
    "skandi_gepi": "Skandináv – gépi",
    "skandi_kezi": "Skandináv – kézi",
    "skandi_kombinalt": "Skandináv – kombinált",
}


class LottoGuiApp:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Magyar Lottó AI – GUI")
        self.root.geometry("1180x780")

        self.project_root = Path(__file__).resolve().parents[2]

        self.game_var = StringVar(value="otos")
        self.mode_var = StringVar(value="compare")
        self.csv_var = StringVar(value=str(self.project_root / DEFAULT_CSV["otos"]))
        self.ticket_count_var = StringVar(value="3")
        self.strategy_var = StringVar(value="diverzifikalt")

        self.status_var = StringVar(value="Készen áll.")
        self._status_base_text = "Elemzés folyamatban"
        self._status_dots = 0
        self._animation_job: str | None = None
        self._is_running = False

        self.latest_result: dict[str, Any] | None = None
        self.latest_source_info: dict[str, Any] | None = None
        self.ticket_window: Toplevel | None = None
        self.ticket_listbox = None

        self._build_ui()
        self._bind_events()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=BOTH, expand=True)

        controls = ttk.LabelFrame(main, text="Beállítások", padding=10)
        controls.pack(fill="x")

        ttk.Label(controls, text="Játék:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        self.game_combo = ttk.Combobox(
            controls,
            textvariable=self.game_var,
            state="readonly",
            width=22,
            values=list(GAME_LABELS.keys()),
        )
        self.game_combo.grid(row=0, column=1, sticky="w", pady=6)

        ttk.Label(controls, text="Mód:").grid(row=0, column=2, sticky="w", padx=(18, 8), pady=6)
        self.mode_combo = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            state="readonly",
            width=28,
            values=list(MODEL_LABELS.keys()),
        )
        self.mode_combo.grid(row=0, column=3, sticky="w", pady=6)

        ttk.Label(controls, text="Szelvények:").grid(row=0, column=4, sticky="w", padx=(18, 8), pady=6)
        ttk.Spinbox(
            controls,
            from_=1,
            to=20,
            textvariable=self.ticket_count_var,
            width=6,
        ).grid(row=0, column=5, sticky="w", pady=6)

        ttk.Label(controls, text="Stratégia:").grid(row=0, column=6, sticky="w", padx=(18, 8), pady=6)
        ttk.Combobox(
            controls,
            textvariable=self.strategy_var,
            state="readonly",
            width=16,
            values=["diverzifikalt", "top"],
        ).grid(row=0, column=7, sticky="w", pady=6)

        ttk.Label(controls, text="CSV fájl:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        self.csv_entry = ttk.Entry(controls, textvariable=self.csv_var, width=92)
        self.csv_entry.grid(row=1, column=1, columnspan=7, sticky="ew", pady=6)

        ttk.Button(controls, text="Tallózás", command=self._browse_csv).grid(
            row=1, column=8, padx=(8, 0), pady=6
        )

        self.run_btn = ttk.Button(controls, text="Futtatás", command=self._run_clicked)
        self.run_btn.grid(row=0, column=8, padx=(8, 0), pady=6)

        self.tickets_btn = ttk.Button(
            controls,
            text="Ajánlott szelvények",
            command=self._show_ticket_window,
            state="disabled",
        )
        self.tickets_btn.grid(row=0, column=9, padx=(8, 0), pady=6)

        controls.columnconfigure(7, weight=1)

        top_pane = ttk.Panedwindow(main, orient="horizontal")
        top_pane.pack(fill=BOTH, expand=True, pady=(12, 0))

        left = ttk.LabelFrame(top_pane, text="Összefoglaló", padding=10)
        right = ttk.LabelFrame(top_pane, text="Részletes kimenet", padding=10)

        top_pane.add(left, weight=1)
        top_pane.add(right, weight=2)

        self.summary_text = ScrolledText(left, wrap="word", height=18)
        self.summary_text.pack(fill=BOTH, expand=True)

        self.output_text = ScrolledText(right, wrap="word", height=18)
        self.output_text.pack(fill=BOTH, expand=True)

        status_bar = ttk.Frame(main)
        status_bar.pack(fill="x", pady=(8, 0))

        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=220)
        self.progress.pack(side=RIGHT, padx=(12, 0))

        ttk.Label(status_bar, textvariable=self.status_var).pack(side=LEFT)

    def _bind_events(self) -> None:
        self.game_combo.bind("<<ComboboxSelected>>", self._on_game_changed)

    def _on_game_changed(self, _event: Any = None) -> None:
        rel = DEFAULT_CSV.get(self.game_var.get())
        if rel:
            self.csv_var.set(str(self.project_root / rel))

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="CSV kiválasztása",
            filetypes=[("CSV fájl", "*.csv"), ("Minden fájl", "*.*")],
        )
        if path:
            self.csv_var.set(path)

    def _run_clicked(self) -> None:
        csv_path = Path(self.csv_var.get().strip())
        jatek = self.game_var.get()

        try:
            source_info = ensure_csv_available(
                jatek=jatek,
                local_path=csv_path,
                force_refresh=True,
            )
        except Exception as exc:
            messagebox.showerror("CSV hiba", str(exc))
            return

        resolved_csv_path = Path(source_info["csv_path"])
        self.latest_source_info = source_info

        self.run_btn.configure(state="disabled")
        self.tickets_btn.configure(state="disabled")
        self.summary_text.delete("1.0", END)
        self.output_text.delete("1.0", END)

        self._start_animation()

        threading.Thread(
            target=self._run_pipeline_safe,
            args=(resolved_csv_path,),
            daemon=True,
        ).start()

    def _start_animation(self) -> None:
        self._is_running = True
        self._status_dots = 0
        self.progress.start(12)
        self._tick_status()

    def _stop_animation(self, final_text: str) -> None:
        self._is_running = False
        if self._animation_job is not None:
            try:
                self.root.after_cancel(self._animation_job)
            except Exception:
                pass
            self._animation_job = None
        self.progress.stop()
        self.status_var.set(final_text)

    def _tick_status(self) -> None:
        if not self._is_running:
            return

        self._status_dots = (self._status_dots + 1) % 4
        self.status_var.set(self._status_base_text + "." * self._status_dots)
        self._animation_job = self.root.after(400, self._tick_status)

    def _run_pipeline_safe(self, csv_path: Path) -> None:
        try:
            result = self._run_pipeline(csv_path)
            self.root.after(0, lambda: self._show_result(result))
        except Exception as exc:
            details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.root.after(0, lambda: self._show_error(details))

    def _run_pipeline(self, csv_path: Path) -> dict[str, Any]:
        mode = self.mode_var.get()
        jatek = self.game_var.get()
        ticket_count = max(1, int(self.ticket_count_var.get() or "1"))
        strategy = self.strategy_var.get() or "diverzifikalt"

        if jatek == "skandi_kombinalt":
            result = run_skandi_combined_pipeline(csv_path=csv_path, mode=mode)
            prediction = (
                result.get("prediction")
                or result.get("models", {}).get("skandi_kombinalt", {}).get("prediction")
            )
            if prediction:
                result["ticket_bundle"] = generate_tickets_from_prediction(
                    prediction,
                    jatek=jatek,
                    ticket_count=ticket_count,
                    strategy=strategy,
                )
            return result

        if mode == "compare":
            result = run_compare_pipeline(csv_path=csv_path, jatek=jatek)
            pred = (result.get("models", {}).get("ensemble_avg", {}) or {}).get("prediction")
        elif mode == "random_forest":
            result = run_random_forest_pipeline(csv_path=csv_path, jatek=jatek)
            pred = result.get("prediction")
        elif mode == "xgboost":
            result = run_xgboost_pipeline(csv_path=csv_path, jatek=jatek)
            pred = result.get("prediction")
        elif mode == "lstm":
            result = run_lstm_pipeline(csv_path=csv_path, jatek=jatek)
            pred = result.get("prediction")
        else:
            raise ValueError(f"Ismeretlen mód: {mode}")

        if pred:
            result["ticket_bundle"] = generate_tickets_from_prediction(
                pred,
                jatek=jatek,
                ticket_count=ticket_count,
                strategy=strategy,
            )

        return result

    def _show_result(self, result: dict[str, Any]) -> None:
        self.latest_result = result
        self.run_btn.configure(state="normal")
        self.tickets_btn.configure(state="normal")

        source_info = self.latest_source_info or {}
        if source_info.get("internet_ok") is False:
            self._stop_animation("Futtatás kész. Offline mód: helyi CSV használva.")
        elif source_info.get("source_mode") == "fallback_local":
            self._stop_animation("Futtatás kész. Online frissítés nem sikerült, helyi CSV használva.")
        else:
            self._stop_animation("Futtatás kész.")

        self.summary_text.insert("1.0", self._format_summary(result))
        self.output_text.insert("1.0", self._format_details(result))

    def _show_error(self, details: str) -> None:
        self.run_btn.configure(state="normal")
        self.tickets_btn.configure(state="disabled")
        self._stop_animation("Hiba történt.")
        self.output_text.insert("1.0", details)
        messagebox.showerror(
            "Futtatási hiba",
            "A pipeline hibára futott. A részletek a jobb oldali panelen vannak.",
        )

    def _extract_tickets(self) -> list[list[int]]:
        if not self.latest_result:
            return []

        bundle = self.latest_result.get("ticket_bundle") or {}
        tickets = bundle.get("tickets") or []
        return tickets

    def _copy_selected_ticket(self) -> None:
        if not self.ticket_window or self.ticket_listbox is None:
            return

        selected = self.ticket_listbox.curselection()
        if not selected:
            return

        idx = selected[0]
        line = self.ticket_listbox.get(idx)

        parts = line.split(":", 1)
        text = parts[1].strip() if len(parts) == 2 else line.strip()

        self.ticket_window.clipboard_clear()
        self.ticket_window.clipboard_append(text)
        self.ticket_window.update()

    def _copy_all_tickets(self, tickets: list[list[int]]) -> None:
        if not self.ticket_window:
            return

        plain = []
        for ticket in tickets:
            plain.append(", ".join(str(n) for n in ticket))

        text = "\n".join(plain)
        self.ticket_window.clipboard_clear()
        self.ticket_window.clipboard_append(text)
        self.ticket_window.update()

    def _toggle_ticket_window_topmost(self) -> None:
        if not self.ticket_window:
            return

        current = bool(self.ticket_window.attributes("-topmost"))
        self.ticket_window.attributes("-topmost", not current)

    def _show_ticket_window(self) -> None:
        tickets = self._extract_tickets()
        if not tickets:
            messagebox.showinfo("Nincs szelvény", "Még nincs megjeleníthető ajánlott szelvény.")
            return

        if self.ticket_window is not None:
            try:
                self.ticket_window.destroy()
            except Exception:
                pass
            self.ticket_window = None

        self.ticket_window = Toplevel(self.root)
        self.ticket_window.title("Ajánlott szelvények")
        self.ticket_window.geometry("430x360")
        self.ticket_window.attributes("-topmost", True)
        self.ticket_window.resizable(True, True)

        container = ttk.Frame(self.ticket_window, padding=12)
        container.pack(fill=BOTH, expand=True)

        ttk.Label(
            container,
            text="Ajánlott szelvények",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(
            container,
            text="A sorok külön kijelölhetők. Az ablak induláskor mindig felül van.",
        ).pack(anchor="w", pady=(0, 8))

        list_frame = ttk.Frame(container)
        list_frame.pack(fill=BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        scrollbar.pack(side=RIGHT, fill="y")

        import tkinter as tk

        self.ticket_listbox = tk.Listbox(
            list_frame,
            activestyle="dotbox",
            exportselection=False,
            selectmode="browse",
            font=("Consolas", 11),
        )
        self.ticket_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        self.ticket_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.ticket_listbox.yview)

        for i, ticket in enumerate(tickets, start=1):
            ticket_text = ", ".join(str(n) for n in ticket)
            self.ticket_listbox.insert(END, f"{i}. szelvény: {ticket_text}")

        if tickets:
            self.ticket_listbox.selection_set(0)
            self.ticket_listbox.activate(0)

        btn_row = ttk.Frame(container)
        btn_row.pack(fill="x", pady=(10, 0))

        ttk.Button(
            btn_row,
            text="Kijelölt másolása",
            command=self._copy_selected_ticket,
        ).pack(side=LEFT)

        ttk.Button(
            btn_row,
            text="Összes másolása",
            command=lambda: self._copy_all_tickets(tickets),
        ).pack(side=LEFT, padx=(8, 0))

        ttk.Button(
            btn_row,
            text="Mindig felül ki/be",
            command=self._toggle_ticket_window_topmost,
        ).pack(side=LEFT, padx=(8, 0))

    def _format_summary(self, result: dict[str, Any]) -> str:
        lines = [
            f"Játék: {GAME_LABELS.get(self.game_var.get(), self.game_var.get())}",
            f"Mód: {MODEL_LABELS.get(self.mode_var.get(), self.mode_var.get())}",
            f"Beolvasott rekordok: {result.get('imported_record_count', '-')}",
            f"Hibás sorok: {result.get('error_count', '-')}",
        ]

        source_info = self.latest_source_info or {}
        if source_info:
            lines.append(f"CSV forrásmód: {source_info.get('source_mode', '-')}")
            lines.append(f"Internet elérhető: {'igen' if source_info.get('internet_ok') else 'nem'}")
            lines.append(f"Forrás üzenet: {source_info.get('message', '-')}")

        if "best_model" in result:
            lines.append(f"Legjobb modell: {result.get('best_model')}")

        scoreboard = result.get("scoreboard") or result.get("ranking") or []
        if scoreboard:
            lines += ["", "Rangsor:"]
            for row in scoreboard:
                if isinstance(row, dict) and "avg_hit_at_5" in row:
                    lines.append(
                        f"- {row.get('modell')}: "
                        f"hit@5={row.get('avg_hit_at_5')}, "
                        f"hit@10={row.get('avg_hit_at_10')}, "
                        f"any@5={row.get('any_hit_rate_at_5')}"
                    )
                else:
                    lines.append(f"- {row}")

        pred = result.get("prediction")
        if not pred and self.mode_var.get() == "compare":
            pred = (result.get("models", {}).get("ensemble_avg", {}) or {}).get("prediction")
        if not pred:
            pred = result.get("ensemble_prediction")

        if pred:
            lines += ["", f"Ajánlott számok: {pred.get('top_szamok')}"]

        bundle = result.get("ticket_bundle") or {}
        tickets = bundle.get("tickets") or []
        if tickets:
            lines += ["", f"Szelvények ({bundle.get('strategy')}):"]
            for i, ticket in enumerate(tickets, start=1):
                lines.append(f"{i}. {ticket}")

        return "\n".join(lines)

    def _format_details(self, result: dict[str, Any]) -> str:
        return json.dumps(self._normalize_for_json(result), indent=2, ensure_ascii=False)

    def _normalize_for_json(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._normalize_for_json(asdict(value))

        if isinstance(value, dict):
            return {str(k): self._normalize_for_json(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [self._normalize_for_json(v) for v in value]

        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()

            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass

        return value

    def run(self) -> None:
        self.root.mainloop()