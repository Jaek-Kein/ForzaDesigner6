from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget, QSplitter
)

from fd6.gui.widgets import ImageView


class PreviewPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal, self)
        self.source_view = ImageView("Source", self)
        self.preview_view = ImageView("Preview", self)
        splitter.addWidget(self.source_view)
        splitter.addWidget(self.preview_view)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, stretch=1)

        info_row = QHBoxLayout()
        self.status_label = QLabel("Idle.", self)
        self.status_label.setStyleSheet("color: #aaa;")
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        info_row.addWidget(self.status_label, stretch=1)
        info_row.addWidget(self.progress, stretch=2)
        layout.addLayout(info_row)

    def set_source(self, path: str | Path) -> None:
        self.source_view.set_path(str(path))
        self.preview_view.clear_image()
        self.progress.setValue(0)
        self._eta_start_time: float | None = None
        self._eta_start_count: int = 0
        self.status_label.setText(
            "Idle — give the FD6 engine a moment to start. "
            "First-shape startup can take anywhere from a few seconds to several "
            "minutes depending on profile (random/mutated samples) and image size."
        )

    def on_progress(self, count: int, total: int, rms: float) -> None:
        pct = int(round(100 * count / max(1, total)))
        self.progress.setValue(min(100, pct))

        now = time.monotonic()
        if not hasattr(self, "_eta_start_time") or self._eta_start_time is None:
            self._eta_start_time = now
            self._eta_start_count = count

        elapsed = now - self._eta_start_time
        done_since_start = count - self._eta_start_count
        remaining = total - count

        if done_since_start > 0 and remaining > 0:
            rate = done_since_start / elapsed  # shapes/sec
            eta_sec = remaining / rate
            if eta_sec < 60:
                eta_str = f"  ETA {eta_sec:.0f}s"
            elif eta_sec < 3600:
                eta_str = f"  ETA {int(eta_sec // 60)}m {int(eta_sec % 60)}s"
            else:
                eta_str = f"  ETA {eta_sec / 3600:.1f}h"
        else:
            eta_str = ""

        self.status_label.setText(f"Shape {count}/{total}   RMS={rms:.2f}{eta_str}")

    def on_preview(self, arr) -> None:
        self.preview_view.set_numpy(arr)

    def reset(self) -> None:
        self.progress.setValue(0)
        self.status_label.setText("Idle.")
        self.source_view.clear_image()
        self.preview_view.clear_image()
