from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QImageReader
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .detect import Settings
from .export import ExportOptions
from .media import Cancelled, MediaError, scan
from .pipeline import process


APP_NAME = "AI Video Merger Agent"
STYLE = """
QWidget { background: #11151d; color: #edf1f8; font: 14px 'Segoe UI'; }
QLabel#title { font-size: 28px; font-weight: 700; }
QLabel#muted, QLabel#path { color: #aab7ca; }
QLabel#metric { color: #a9c8ff; font-weight: 600; padding: 0 12px; }
QFrame#panel, QTabWidget::pane { border: 1px solid #2c374a; border-radius: 8px; }
QPushButton { background: #202b3d; border: 1px solid #374963; border-radius: 6px;
              padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background: #2d3d57; }
QPushButton#start { color: #101b2c; background: #9fc4ff; border-color: #9fc4ff; }
QPushButton#start:hover { background: #bfd8ff; }
QPushButton:disabled, QPushButton#start:disabled { color: #778396; background: #1a2230; }
QTabBar::tab { background: #1b2431; color: #aab7ca; padding: 9px 18px; }
QTabBar::tab:selected { background: #28364d; color: #edf1f8; }
QTableWidget, QPlainTextEdit { background: #151c27; alternate-background-color: #192332;
                               border: 0; selection-background-color: #36557f; }
QHeaderView::section { background: #222e41; color: #cbd6e7; padding: 8px; border: 0; }
QProgressBar { border: 1px solid #33445f; background: #1b2332; border-radius: 5px;
               text-align: center; min-height: 19px; }
QProgressBar::chunk { background: #4878b8; border-radius: 4px; }
QDoubleSpinBox, QSpinBox, QComboBox { background: #202b3d; padding: 5px; border: 1px solid #374963; }
"""


class Worker(QObject):
    progress = Signal(str, int)
    log = Signal(str)
    stats = Signal(int, float)
    result = Signal(object)
    error = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, action):
        super().__init__()
        self.action = action
        self.cancel = Event()

    @Slot()
    def execute(self):
        try:
            self.result.emit(self.action(self))
        except Cancelled:
            self.cancelled.emit()
        except MediaError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.log.emit(traceback.format_exc())
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


def button(text, callback, name=""):
    widget = QPushButton(text)
    widget.setObjectName(name)
    widget.clicked.connect(callback)
    return widget


def label(text, name=""):
    widget = QLabel(text)
    widget.setObjectName(name)
    return widget


def table(headers):
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.verticalHeader().hide()
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    widget.setAlternatingRowColors(True)
    widget.setShowGrid(False)
    widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    return widget


def fill(widget, rows):
    widget.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            item = QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            widget.setItem(i, j, item)


def elapsed(seconds):
    milliseconds = round(seconds * 1000)
    minutes, remainder = divmod(milliseconds, 60_000)
    return f"{minutes:02d}:{remainder/1000:06.3f}"


class Preferences(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Processing settings")
        layout = QFormLayout(self)
        self.controls = {}
        for key, title, low, high, step in [
            ("minimum_seconds", "Minimum repeated sequence (seconds)", .5, 30., .25),
            ("hash_similarity", "Perceptual hash similarity", .5, 1., .01),
            ("tile_similarity", "Spatial similarity", .5, 1., .01),
            ("audio_correlation", "Audio correlation", .8, 1., .01),
        ]:
            control = QDoubleSpinBox()
            control.setRange(low, high)
            control.setSingleStep(step)
            control.setValue(getattr(parent.detector, key))
            self.controls[key] = control
            layout.addRow(title, control)
        self.resolution = QComboBox()
        resolutions = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160),
                       (720, 1280), (1080, 1920)]
        for width, height in resolutions:
            self.resolution.addItem(f"{width} × {height}", (width, height))
        self.resolution.setCurrentIndex(resolutions.index((parent.options.width, parent.options.height)))
        layout.addRow("Output resolution", self.resolution)
        self.fps = QComboBox()
        for fps in (24, 25, 30, 50, 60):
            self.fps.addItem(f"{fps} FPS", fps)
        self.fps.setCurrentIndex(self.fps.findData(parent.options.fps))
        layout.addRow("Output frame rate", self.fps)
        self.encoder = QComboBox()
        for text, key in [("CPU · x264", "libx264"), ("NVIDIA · NVENC", "h264_nvenc"),
                          ("Intel · Quick Sync", "h264_qsv"), ("AMD · AMF", "h264_amf")]:
            self.encoder.addItem(text, key)
        self.encoder.setCurrentIndex(self.encoder.findData(parent.options.encoder))
        layout.addRow("H.264 encoder", self.encoder)
        self.transitions = QCheckBox("Crossfade eligible quiet clip joins")
        self.transitions.setChecked(parent.options.transitions)
        layout.addRow(self.transitions)
        self.revisions = QCheckBox("Prefer newest cumulative revision")
        self.revisions.setChecked(parent.options.prefer_latest_revision)
        self.revisions.setToolTip("For progressively longer exports of the same story. "
                                  "The newest version can replace an earlier alternate ending.")
        layout.addRow(self.revisions)
        self.watermark_path = parent.options.watermark_path
        self.watermark_label = label(Path(self.watermark_path).name if self.watermark_path else "Disabled", "muted")
        self.watermark_label.setToolTip(self.watermark_path)
        layout.addRow("Channel watermark", self.watermark_label)
        layout.addRow(button("Choose logo image", self.pick_watermark),
                      button("Clear watermark", self.clear_watermark))
        self.watermark_corner = QComboBox()
        for corner in ("bottom-right", "bottom-left", "top-right", "top-left"):
            self.watermark_corner.addItem(corner.replace("-", " ").title(), corner)
        self.watermark_corner.setCurrentIndex(self.watermark_corner.findData(parent.options.watermark_corner))
        layout.addRow("Logo corner", self.watermark_corner)
        self.watermark_width = QSpinBox()
        self.watermark_width.setRange(4, 35)
        self.watermark_width.setSuffix("%")
        self.watermark_width.setValue(parent.options.watermark_width)
        layout.addRow("Logo size", self.watermark_width)
        self.watermark_opacity = QSpinBox()
        self.watermark_opacity.setRange(10, 100)
        self.watermark_opacity.setSuffix("%")
        self.watermark_opacity.setValue(parent.options.watermark_opacity)
        layout.addRow("Logo opacity", self.watermark_opacity)
        self.model = parent.model
        self.model_label = label(str(self.model or "Disabled"), "muted")
        self.model_label.setWordWrap(True)
        layout.addRow("Local Whisper model", self.model_label)
        layout.addRow(button("Select model folder", self.pick_model), button("Disable speech", self.clear_model))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def pick_watermark(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose channel watermark", self.watermark_path,
                                              "Image files (*.png *.jpg *.jpeg *.webp)")
        if path:
            reader = QImageReader(path)
            if not reader.canRead():
                QMessageBox.warning(self, APP_NAME, "The selected image could not be read.")
                return
            self.watermark_path = path
            self.watermark_label.setText(Path(path).name)
            self.watermark_label.setToolTip(path)

    def clear_watermark(self):
        self.watermark_path = ""
        self.watermark_label.setText("Disabled")
        self.watermark_label.setToolTip("")

    def pick_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select a local CTranslate2 Whisper model")
        if path:
            self.model = Path(path)
            self.model_label.setText(path)

    def clear_model(self):
        self.model = None
        self.model_label.setText("Disabled")


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1030, 710)
        self.setMinimumSize(810, 550)
        self.store = QSettings("AI Video Merger Agent", "Desktop")
        self.destination = Path(self.store.value("output", str(Path.home() / "Videos" / APP_NAME)))
        self.history_paths = []
        self.folder = None
        self.clips = []
        self.warnings = []
        self.detector = Settings()
        self.options = ExportOptions(
            watermark_path=self.store.value("watermark/path", "", type=str),
            watermark_corner=self.store.value("watermark/corner", "bottom-right", type=str),
            watermark_width=self.store.value("watermark/width", 12, type=int),
            watermark_opacity=self.store.value("watermark/opacity", 85, type=int),
        )
        self.model = None
        self.thread = None
        self.worker = None

        page = QWidget()
        self.setCentralWidget(page)
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(14)
        head = QHBoxLayout()
        head.addWidget(label(APP_NAME, "title"))
        head.addStretch()
        self.preferences = button("Settings", self.settings)
        head.addWidget(self.preferences)
        root.addLayout(head)

        panel = QFrame(objectName="panel")
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(14, 12, 14, 12)
        row = QHBoxLayout()
        self.input = button("Select Input Folder", self.select_input)
        self.input_path = label("No folder selected", "path")
        self.input_path.setWordWrap(True)
        row.addWidget(self.input)
        row.addWidget(self.input_path, 1)
        self.clip_count = label("0 clips", "metric")
        row.addWidget(self.clip_count)
        controls.addLayout(row)
        row = QHBoxLayout()
        self.output = button("Change Output Folder", self.select_output)
        self.output_path = label(str(self.destination), "path")
        self.output_path.setWordWrap(True)
        row.addWidget(self.output)
        row.addWidget(self.output_path, 1)
        row.addWidget(button("Open Output Folder", self.open_output))
        controls.addLayout(row)
        root.addWidget(panel)

        self.tabs = QTabWidget()
        self.files = table(["Imported video", "Duration", "Resolution", "Frame rate"])
        self.history = table(["Completed run · double-click to open", "Duration", "Removed"])
        self.history.cellDoubleClicked.connect(self.open_history)
        self.logs = QPlainTextEdit(readOnly=True)
        self.logs.setMaximumBlockCount(4000)
        self.tabs.addTab(self.files, "Clips")
        self.tabs.addTab(self.history, "History")
        self.tabs.addTab(self.logs, "Diagnostics")
        root.addWidget(self.tabs, 1)

        status = QHBoxLayout()
        self.stage = label("Select an input folder", "muted")
        self.duplicates = label("0 removals · 0.00 s", "metric")
        status.addWidget(self.stage, 1)
        status.addWidget(self.duplicates)
        root.addLayout(status)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        root.addWidget(self.progress)
        actions = QHBoxLayout()
        actions.addStretch()
        self.cancel_button = button("Cancel", self.stop)
        self.start = button("Start Automatic Merging", self.merge, "start")
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.start)
        root.addLayout(actions)
        self.set_busy(False)
        self.refresh_history()

    def set_busy(self, busy):
        for widget in (self.input, self.output, self.preferences):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.start.setEnabled(not busy and bool(self.clips))

    def launch(self, action, on_result):
        self.set_busy(True)
        self.thread = QThread(self)
        self.worker = Worker(action)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.execute)
        self.worker.result.connect(on_result)
        self.worker.progress.connect(self.update_status)
        self.worker.log.connect(self.write_log)
        self.worker.stats.connect(self.update_stats)
        self.worker.error.connect(self.show_error)
        self.worker.cancelled.connect(self.show_cancelled)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.finished)
        self.thread.start()

    @Slot()
    def finished(self):
        self.thread.deleteLater()
        self.thread = self.worker = None
        self.set_busy(False)

    def select_input(self):
        picked = QFileDialog.getExistingDirectory(self, "Select input folder", str(self.folder or Path.home()))
        if picked:
            self.folder = Path(picked)
            self.input_path.setText(picked)
            self.clips = []
            self.warnings = []
            self.clip_count.setText("0 clips")
            fill(self.files, [])
            self.tabs.setCurrentWidget(self.files)
            self.update_stats(0, 0.)
            self.update_status("Validating clips with FFprobe", -1)
            folder = self.folder
            self.launch(lambda worker: scan(folder, worker.cancel, worker.log.emit), self.imported)

    @Slot(object)
    def imported(self, result):
        self.clips, self.warnings = result
        fill(self.files, [(c.path.name, elapsed(c.duration), f"{c.width} × {c.height}", f"{c.fps:.2f} fps")
                          for c in self.clips])
        self.clip_count.setText(f"{len(self.clips)} clips")
        suffix = f" · {len(self.warnings)} skipped" if self.warnings else ""
        self.update_status(("Ready to merge" if self.clips else "No playable clips found") + suffix, 0)

    def select_output(self):
        picked = QFileDialog.getExistingDirectory(self, "Select output folder", str(self.destination.parent))
        if picked:
            self.destination = Path(picked)
            self.output_path.setText(picked)
            self.store.setValue("output", picked)
            self.refresh_history()

    def settings(self):
        dialog = Preferences(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.detector = Settings(**{key: control.value() for key, control in dialog.controls.items()})
            width, height = dialog.resolution.currentData()
            self.options = ExportOptions(width, height, dialog.fps.currentData(),
                                         dialog.encoder.currentData(), dialog.transitions.isChecked(),
                                         dialog.watermark_path, dialog.watermark_corner.currentData(),
                                         dialog.watermark_width.value(), dialog.watermark_opacity.value(),
                                         dialog.revisions.isChecked())
            self.store.setValue("watermark/path", self.options.watermark_path)
            self.store.setValue("watermark/corner", self.options.watermark_corner)
            self.store.setValue("watermark/width", self.options.watermark_width)
            self.store.setValue("watermark/opacity", self.options.watermark_opacity)
            self.model = dialog.model

    def merge(self):
        clips, warnings = self.clips[:], self.warnings[:]
        destination, detector, options, model = self.destination, self.detector, self.options, self.model
        self.update_stats(0, 0.)
        self.launch(lambda worker: process(clips, warnings, destination, detector, options, model, worker.cancel,
                                           worker.progress.emit, worker.log.emit, worker.stats.emit), self.exported)

    @Slot(object)
    def exported(self, folder):
        self.write_log(f"Saved {folder / 'merged.mp4'}")
        self.refresh_history()
        self.tabs.setCurrentWidget(self.history)

    def refresh_history(self):
        self.history_paths = []
        rows = []
        if self.destination.is_dir():
            for folder in sorted(self.destination.iterdir(), reverse=True):
                report = folder / "processing-report.json"
                if not (folder / "merged.mp4").is_file() or not report.is_file():
                    continue
                try:
                    data = json.loads(report.read_text(encoding="utf-8"))
                    completed = datetime.fromisoformat(data["completed"]).strftime("%Y-%m-%d %H:%M:%S")
                    revisions = len(data.get("superseded_revisions", []))
                    removed = (f"{revisions} earlier versions" if revisions else
                               f'{data["removed_count"]} · {data["removed_seconds"]:.2f} s')
                    rows.append((f"{completed}  ·  {folder.name}", elapsed(data["duration"]), removed))
                    self.history_paths.append(folder)
                except (OSError, ValueError, TypeError, KeyError):
                    continue
        fill(self.history, rows)

    def open_history(self, row, _column):
        if 0 <= row < len(self.history_paths):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.history_paths[row])))

    def open_output(self):
        try:
            self.destination.mkdir(parents=True, exist_ok=True)
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.destination))):
                raise OSError("Windows could not open the output folder.")
        except OSError as exc:
            self.show_error(str(exc))

    @Slot(str, int)
    def update_status(self, stage, percent):
        self.stage.setText(stage)
        self.progress.setRange(0, 100 if percent >= 0 else 0)
        if percent >= 0:
            self.progress.setValue(percent)

    @Slot(int, float)
    def update_stats(self, count, seconds):
        self.duplicates.setText(f"{count} removals · {seconds:.2f} s")

    @Slot(str)
    def write_log(self, text):
        self.logs.appendPlainText(f"{datetime.now():%H:%M:%S}  {text}")

    @Slot(str)
    def show_error(self, text):
        self.write_log(text)
        self.update_status("Processing failed", 0)
        self.tabs.setCurrentWidget(self.logs)
        QMessageBox.critical(self, APP_NAME, text)

    @Slot()
    def show_cancelled(self):
        self.update_status("Cancelled", 0)
        self.write_log("Operation cancelled.")

    def stop(self):
        if self.worker:
            self.worker.cancel.set()
            self.stage.setText("Cancelling…")
            self.cancel_button.setEnabled(False)

    def closeEvent(self, event):
        if self.thread:
            self.stop()
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    sys.exit(app.exec())

