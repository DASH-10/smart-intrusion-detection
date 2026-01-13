from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - optional UI dependency
    tk = None
    ttk = None


@dataclass
class PanelState:
    running: bool = True
    enable_open_vocab: bool = True
    save_video: bool = False
    save_screenshots: bool = True
    tracker_type: str = "centroid"
    source: str = ""


class ControlPanel:
    """Tiny Tkinter control panel to toggle runtime flags without a web UI."""

    def __init__(self, initial: Optional[PanelState] = None):
        self.state = initial or PanelState()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if tk is None or ttk is None:
            raise ImportError("Tkinter not available; cannot start control panel.")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        root = tk.Tk()
        root.title("Smart Intrusion Control")

        running_var = tk.BooleanVar(value=self.state.running)
        ov_var = tk.BooleanVar(value=self.state.enable_open_vocab)
        video_var = tk.BooleanVar(value=self.state.save_video)
        shots_var = tk.BooleanVar(value=self.state.save_screenshots)
        tracker_var = tk.StringVar(value=self.state.tracker_type)
        source_var = tk.StringVar(value=self.state.source)

        def sync_state():
            self.state.running = running_var.get()
            self.state.enable_open_vocab = ov_var.get()
            self.state.save_video = video_var.get()
            self.state.save_screenshots = shots_var.get()
            self.state.tracker_type = tracker_var.get()
            self.state.source = source_var.get()

        def toggle_running():
            running_var.set(not running_var.get())
            sync_state()

        tk.Label(root, text="Controls").grid(row=0, column=0, columnspan=2, pady=4)

        tk.Button(root, text="Start/Stop", command=toggle_running).grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        ttk.Checkbutton(root, text="Enable OpenVocab", variable=ov_var, command=sync_state).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        ttk.Checkbutton(root, text="Save Video", variable=video_var, command=sync_state).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        ttk.Checkbutton(root, text="Save Screenshots", variable=shots_var, command=sync_state).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )

        tk.Label(root, text="Tracker").grid(row=5, column=0, sticky="w", padx=4, pady=2)
        tracker_menu = ttk.Combobox(root, textvariable=tracker_var, values=["centroid", "deepsort", "bytetrack"])
        tracker_menu.grid(row=5, column=1, sticky="ew", padx=4, pady=2)
        tracker_menu.bind("<<ComboboxSelected>>", lambda _evt: sync_state())

        tk.Label(root, text="Source").grid(row=6, column=0, sticky="w", padx=4, pady=2)
        tk.Entry(root, textvariable=source_var).grid(row=6, column=1, sticky="ew", padx=4, pady=2)

        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.mainloop()

