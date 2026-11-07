"""The instructions panel shown on every page."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QTextBrowser

HELP_DIR = Path(__file__).resolve().parent.parent / "help"


def help_markdown(key: str) -> str:
    try:
        return (HELP_DIR / f"{key}.md").read_text(encoding="utf-8")
    except OSError:
        return f"# Help\n\nNo help text found for '{key}'."


class HelpPanel(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("helpPanel")
        self.setOpenExternalLinks(True)
        self.setMinimumWidth(300)

    def show_help(self, key: str) -> None:
        self.setMarkdown(help_markdown(key))
        self.verticalScrollBar().setValue(0)
