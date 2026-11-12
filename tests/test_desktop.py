import time

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog

from merger.gui import Preferences, Window
from test_merger import binaries, footage, make_clip


def test_desktop_folder_to_export(tmp_path, monkeypatch, binaries):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr("merger.gui.QSettings",
                        lambda *_: QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    sources = tmp_path / "sources"
    sources.mkdir()
    make_clip(sources, "clip1", [footage(4)], mute=True)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_: str(sources))
    window = Window()
    window.destination = tmp_path / "output"
    window.show()

    def wait_for_worker():
        deadline = time.monotonic() + 60
        while window.thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert window.thread is None

    errors = []
    monkeypatch.setattr(window, "show_error", errors.append)
    try:
        preferences = Preferences(window)
        assert preferences.resolution.currentData() == (1920, 1080)
        assert preferences.watermark_corner.currentData() == "bottom-right"
        assert preferences.watermark_width.value() == 12
        preferences.close()
        window.input.click()
        wait_for_worker()
        assert window.files.rowCount() == 1
        assert window.start.isEnabled()
        window.start.click()
        wait_for_worker()
        assert not errors
        assert window.progress.value() == 100
        assert window.history.rowCount() == 1
        assert (window.history_paths[0] / "merged.mp4").is_file()
    finally:
        window.close()

