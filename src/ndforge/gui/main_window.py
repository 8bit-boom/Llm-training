"""The guided main window: sidebar with the five steps, a page area and the instructions panel."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QStackedWidget,
)

from ..core.project import Project
from ..core.settings import Settings
from .help_panel import HelpPanel
from .pages import DataPage, PlaceholderPage, WelcomePage

STYLE = """
QLabel#pageTitle { font-size: 22px; font-weight: 600; padding: 4px 0 8px 0; }
QLabel#note { font-style: italic; padding-bottom: 8px; }
QLabel#error { color: #c0392b; }
QLabel#cardTitle { font-size: 16px; font-weight: 600; }
QFrame#card { border: 1px solid palette(mid); border-radius: 8px; }
QPushButton#primary { padding: 8px 18px; font-weight: 600; min-width: 16em; }
QListWidget#steps { font-size: 14px; }
QListWidget#steps::item { padding: 10px 8px; }
"""

STEP_DEFS = [
    ("welcome", "Welcome", None),
    ("data", "1 · Get world data", "data"),
    ("dataset", "2 · Make training data", "dataset"),
    ("train", "3 · Train", "train"),
    ("test", "4 · Test", "test"),
    ("use", "5 · Use in nd-world", "use"),
]


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None, ask=None):
        super().__init__()
        self.settings = settings or Settings.load()
        self.project: Project | None = None
        self.setWindowTitle("ND Lore Forge")
        self.resize(1250, 780)
        self.setStyleSheet(STYLE)

        self.steps = QListWidget()
        self.steps.setObjectName("steps")
        self.steps.setFixedWidth(230)
        self.stack = QStackedWidget()
        self.help = HelpPanel()
        self.pages = {
            "welcome": WelcomePage(self.settings),
            "data": DataPage(lambda: self.project, ask=ask),
            "dataset": PlaceholderPage("dataset", "Step 2 · Make training data", "Coming in a later version. See the help panel for what this step will do."),
            "train": PlaceholderPage("train", "Step 3 · Train", "Coming in a later version. See the help panel for what this step will do."),
            "test": PlaceholderPage("test", "Step 4 · Test", "Coming in a later version. See the help panel for what this step will do."),
            "use": PlaceholderPage("use", "Step 5 · Use in nd-world", "Coming in a later version. See the help panel for what this step will do."),
        }
        for key, label, _ in STEP_DEFS:
            self.stack.addWidget(self.pages[key])
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            self.steps.addItem(item)
        self.pages["welcome"].project_chosen.connect(self.set_project)
        self.pages["data"].changed.connect(self.update_steps)
        self.steps.currentRowChanged.connect(self._row_changed)

        split = QSplitter()
        split.addWidget(self.steps)
        split.addWidget(self.stack)
        split.addWidget(self.help)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)
        split.setSizes([230, 650, 370])
        self.setCentralWidget(split)

        self.project_label = QLabel("No project open")
        self.statusBar().addWidget(self.project_label, 1)

        bar = self.addToolBar("Main")
        bar.setMovable(False)
        self.help_action = QAction("Help panel", self, checkable=True, checked=self.settings.show_help_panel)
        self.help_action.toggled.connect(self._toggle_help)
        self.expert_action = QAction("Expert mode", self, checkable=True, checked=self.settings.expert_mode)
        self.expert_action.toggled.connect(self._toggle_expert)
        bar.addAction(self.help_action)
        bar.addAction(self.expert_action)
        self.help.setVisible(self.settings.show_help_panel)

        self.update_steps()
        self.steps.setCurrentRow(0)
        self.pages["welcome"].refresh()

    # ----------------------------------------------------------- project
    def set_project(self, project: Project) -> None:
        self.project = project
        self.settings.remember_project(project.folder)
        self.settings.save()
        self.project_label.setText(f"Project: {project.name}  ({project.folder})")
        self.update_steps()
        for page in self.pages.values():
            page.refresh()
        self.go("data")

    def update_steps(self) -> None:
        unlocked = self.project.unlocked() if self.project else {}
        done = self.project.progress() if self.project else {}
        for i, (key, label, flag) in enumerate(STEP_DEFS):
            item = self.steps.item(i)
            if flag is None:
                enabled, text = True, label
            else:
                enabled = unlocked.get(flag, False)
                text = ("✓ " if done.get(flag) else "🔒 " if not enabled else "") + label
            item.setText(text)
            item.setFlags(item.flags() | Qt.ItemIsEnabled if enabled else item.flags() & ~Qt.ItemIsEnabled)

    # --------------------------------------------------------------- nav
    def go(self, key: str) -> None:
        for i, (k, _, _) in enumerate(STEP_DEFS):
            if k == key:
                self.steps.setCurrentRow(i)
                return

    def _row_changed(self, row: int) -> None:
        if row < 0:
            return
        page = self.stack.widget(row)
        self.stack.setCurrentIndex(row)
        page.refresh()
        self.help.show_help(page.help_key)

    def _toggle_help(self, on: bool) -> None:
        self.help.setVisible(on)
        self.settings.show_help_panel = on
        self.settings.save()

    def _toggle_expert(self, on: bool) -> None:
        self.settings.expert_mode = on
        self.settings.save()
