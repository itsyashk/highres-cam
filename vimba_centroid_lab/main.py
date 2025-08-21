"""main.py – entry point to run Vimba Centroid Lab."""
from __future__ import annotations

import sys
from pathlib import Path
import yaml

from PySide6.QtWidgets import QApplication

from .ui_main import MainWindow


def main():
    root = Path(__file__).resolve().parent
    config_path = root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    app = QApplication(sys.argv)
    win = MainWindow(config)
    win.resize(1024, 768)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 