"""Tkinter GUI with draft->revised update visibility."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable


class TranscriberGUI:
    def __init__(
        self,
        on_start: Callable[[str, bool, str], bool],
        on_stop: Callable[[], None],
    ) -> None:
        self.on_start = on_start
        self.on_stop = on_stop

        self.root = tk.Tk()
        self.root.title("高精度日语转写系统（初稿→修正版）")
        self.root.geometry("1020x760")

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self) -> None:
        self.root.mainloop()

    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="输出TXT:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value=str(Path.cwd() / "transcript_output.txt"))
        ttk.Entry(top, textvariable=self.path_var, width=64).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="选择", command=self._choose_path).pack(side=tk.LEFT)

        self.hotword_var = tk.StringVar(value="")
        ttk.Label(top, text="热词词典(可选):").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(top, textvariable=self.hotword_var, width=28).pack(side=tk.LEFT)

        ctrl = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        ctrl.pack(fill=tk.X)

        self.start_btn = ttk.Button(ctrl, text="开始", command=self._start)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(ctrl, text="停止", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=8)

        self.srt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="导出SRT", variable=self.srt_var).pack(side=tk.LEFT, padx=8)

        self.status_var = tk.StringVar(value="状态：未启动")
        ttk.Label(ctrl, textvariable=self.status_var).pack(side=tk.LEFT, padx=18)

        pane = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        f1 = ttk.Labelframe(pane, text="实时初稿（低延迟）")
        self.draft_text = ScrolledText(f1, wrap=tk.WORD, font=("Consolas", 11), height=10)
        self.draft_text.pack(fill=tk.BOTH, expand=True)
        pane.add(f1, weight=1)

        f2 = ttk.Labelframe(pane, text="修正版（高精度覆盖）")
        self.final_text = ScrolledText(f2, wrap=tk.WORD, font=("Consolas", 11), height=18)
        self.final_text.pack(fill=tk.BOTH, expand=True)
        pane.add(f2, weight=1)

    def _choose_path(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if p:
            self.path_var.set(p)

    def _start(self) -> None:
        ok = self.on_start(self.path_var.get().strip(), self.srt_var.get(), self.hotword_var.get().strip())
        if ok:
            self.set_running_ui(True)
            self.status_var.set("状态：运行中")

    def _stop(self) -> None:
        self.on_stop()
        self.set_running_ui(False)
        self.status_var.set("状态：已停止")

    def set_running_ui(self, running: bool) -> None:
        self.start_btn.config(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL if running else tk.DISABLED)

    def show_draft(self, text: str) -> None:
        self.draft_text.insert(tk.END, text + "\n\n")
        self.draft_text.see(tk.END)

    def render_final(self, content: str) -> None:
        self.final_text.delete("1.0", tk.END)
        self.final_text.insert(tk.END, content)
        self.final_text.see(tk.END)

    def set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def _on_close(self) -> None:
        self.on_stop()
        self.root.destroy()
