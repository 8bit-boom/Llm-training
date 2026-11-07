"""The pages of the guided window. Each page has a `help_key` and tells the window when it should refresh."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.paths import default_projects_dir
from ..core.project import Project, ProjectError
from ..core.settings import Settings
from ..ingest.bundle import BundleError, open_bundle


def _title(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("pageTitle")
    lab.setWordWrap(True)
    return lab


def _note(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setObjectName("note")
    return lab


class Page(QWidget):
    help_key = "welcome"

    def refresh(self) -> None:  # called when the page is shown or the project changed
        pass


class WelcomePage(Page):
    help_key = "welcome"
    project_chosen = Signal(object)

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Welcome to ND Lore Forge"))
        lay.addWidget(_note("Teach a small AI model your nd-world world. Start by creating a project (one project = one world)."))
        row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Project name, for example the name of your world")
        self.create_btn = QPushButton("Create a new project")
        self.create_btn.setObjectName("primary")
        self.create_btn.clicked.connect(self._create)
        self.name_edit.returnPressed.connect(self._create)
        row.addWidget(self.name_edit, 1)
        row.addWidget(self.create_btn)
        lay.addLayout(row)
        self.error = QLabel()
        self.error.setObjectName("error")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        open_btn = QPushButton("Open an existing project...")
        open_btn.clicked.connect(self._open_dialog)
        lay.addWidget(open_btn, 0, Qt.AlignLeft)
        lay.addWidget(QLabel("Recent projects"))
        self.recent = QListWidget()
        self.recent.itemActivated.connect(lambda it: self._open(it.data(Qt.UserRole)))
        lay.addWidget(self.recent, 1)

    def refresh(self) -> None:
        self.recent.clear()
        for folder in self.settings.recent_projects:
            if (Path(folder) / "project.toml").exists():
                it = QListWidgetItem(f"{Path(folder).name}   —   {folder}")
                it.setData(Qt.UserRole, folder)
                self.recent.addItem(it)

    def _create(self) -> None:
        try:
            project = Project.create(default_projects_dir(), self.name_edit.text())
        except (ProjectError, OSError) as exc:
            self.error.setText(str(exc))
            return
        self.error.clear()
        self.name_edit.clear()
        self.project_chosen.emit(project)

    def _open_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a project folder", str(default_projects_dir()))
        if folder:
            self._open(folder)

    def _open(self, folder: str) -> None:
        try:
            project = Project.open(folder)
        except ProjectError as exc:
            self.error.setText(str(exc))
            return
        self.error.clear()
        self.project_chosen.emit(project)


class DataPage(Page):
    """Step 1: import a .ndtrain file."""
    help_key = "data"
    changed = Signal()

    def __init__(self, get_project: Callable[[], Project | None], ask: Callable[[str, str], bool] | None = None, parent=None):
        super().__init__(parent)
        self.get_project = get_project
        self.ask = ask or self._ask_box
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Step 1 · Get world data"))
        lay.addWidget(_note("Export a <b>Training bundle</b> from nd-world (Tools → Export &amp; Backup), then choose the file here "
                            "or drag it onto this page. The help panel on the right explains how."))
        self.choose_btn = QPushButton("Choose training file...")
        self.choose_btn.setObjectName("primary")
        self.choose_btn.clicked.connect(self._choose)
        lay.addWidget(self.choose_btn, 0, Qt.AlignLeft)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setObjectName("error")
        lay.addWidget(self.message)
        self.card = QFrame()
        self.card.setObjectName("card")
        cl = QVBoxLayout(self.card)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.setSpacing(8)
        self.world_label = QLabel()
        self.world_label.setObjectName("cardTitle")
        self.contents = QLabel()
        self.contents.setTextFormat(Qt.PlainText)
        self.diff_label = QLabel()
        self.diff_label.setWordWrap(True)
        cl.addWidget(self.world_label)
        cl.addWidget(self.contents)
        cl.addWidget(self.diff_label)
        lay.addWidget(self.card)
        lay.addStretch(1)
        self.refresh()

    @staticmethod
    def _ask_box(title: str, text: str) -> bool:
        return QMessageBox.question(None, title, text) == QMessageBox.Yes

    def refresh(self) -> None:
        project = self.get_project()
        bundle = project.current_bundle() if project else None
        self.card.setVisible(bundle is not None)
        if bundle:
            self.world_label.setText(f"World: {bundle.world_name}")
            self.contents.setText("This project contains:\n• " + "\n• ".join(bundle.summary_lines() or ["(no records)"]))
            self.choose_btn.setText("Choose a newer training file...")

    def _choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a training file", "", "Training bundle (*.ndtrain);;All files (*)")
        if path:
            self.import_path(path)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.import_path(urls[0].toLocalFile())

    def import_path(self, path: str | Path) -> bool:
        """Verify and import a file. Returns True when it became the project's current data."""
        project = self.get_project()
        if project is None:
            self.message.setText("Create or open a project first (Welcome page).")
            return False
        self.message.clear()
        self.diff_label.clear()
        try:
            bundle = open_bundle(path)
            diff, problem = project.check_new_bundle(bundle)
            if problem:
                raise ProjectError(problem)
            if diff is not None and not self.ask("Update the world data?", f"{diff.summary()}\n\nUse this file as the new data?"):
                return False
            project.import_bundle(bundle)
        except (BundleError, ProjectError) as exc:
            self.message.setText(str(exc))
            return False
        self.refresh()
        if diff is not None:
            self.diff_label.setText("Changes from the previous file: " + diff.summary())
        self.changed.emit()
        return True


class PlaceholderPage(Page):
    """Pages for steps that are not built yet; they show what the step will do."""

    def __init__(self, key: str, title: str, text: str, parent=None):
        super().__init__(parent)
        self.help_key = key
        lay = QVBoxLayout(self)
        lay.addWidget(_title(title))
        lay.addWidget(_note(text))
        lay.addStretch(1)
