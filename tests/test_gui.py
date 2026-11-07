from PySide6.QtCore import Qt

from ndforge.core.project import Project
from ndforge.core.settings import Settings
from ndforge.gui.main_window import MainWindow
from sample_bundle import build


def _win(qtbot, answers=None):
    w = MainWindow(Settings(), ask=lambda t, m: True if answers is None else answers.pop(0))
    qtbot.addWidget(w)
    return w


def test_starts_on_welcome_with_steps_locked(qtbot):
    w = _win(qtbot)
    assert w.steps.currentRow() == 0 and "Welcome" in w.help.toPlainText()
    assert all(not (w.steps.item(i).flags() & Qt.ItemIsEnabled) for i in range(2, 6))


def test_create_project_goes_to_step_1_and_import_unlocks_step_2(qtbot, tmp_path):
    w = _win(qtbot)
    page = w.pages["welcome"]
    page.name_edit.setText("Ashfall")
    page.create_btn.click()
    assert w.project and w.project.name == "Ashfall" and w.steps.currentRow() == 1
    assert "Get world data" in w.help.toPlainText()
    assert w.pages["data"].import_path(build(tmp_path / "a.ndtrain"))
    assert "2 entities" in w.pages["data"].contents.text() and w.pages["data"].card.isVisibleTo(w)
    assert w.steps.item(1).text().startswith("✓") and w.steps.item(2).flags() & Qt.ItemIsEnabled
    assert not (w.steps.item(3).flags() & Qt.ItemIsEnabled)


def test_bad_file_shows_plain_message(qtbot, tmp_path):
    w = _win(qtbot)
    w.set_project(Project.create(tmp_path / "p", "X"))
    bad = tmp_path / "bad.txt"
    bad.write_text("nope")
    assert not w.pages["data"].import_path(bad)
    assert "not a training bundle" in w.pages["data"].message.text()


def test_update_asks_and_shows_changes(qtbot, tmp_path):
    w = _win(qtbot, answers=[False, True])
    w.set_project(Project.create(tmp_path / "p", "X"))
    data = w.pages["data"]
    assert data.import_path(build(tmp_path / "1.ndtrain", 1))
    new = build(tmp_path / "2.ndtrain", 2)
    assert not data.import_path(new)          # declined
    assert data.import_path(new)              # accepted
    assert "1 new, 1 changed, 1 removed" in data.diff_label.text()


def test_help_panel_toggle_and_recent_projects(qtbot, tmp_path):
    w = _win(qtbot)
    w.set_project(Project.create(tmp_path / "p", "Recent One"))
    w.help_action.setChecked(False)
    assert not w.help.isVisibleTo(w)
    w.pages["welcome"].refresh()
    assert w.pages["welcome"].recent.count() == 1
