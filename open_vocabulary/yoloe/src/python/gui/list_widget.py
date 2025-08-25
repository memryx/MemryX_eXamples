from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QCheckBox,
    QSpacerItem, QSizePolicy, QPushButton, QLabel, QMessageBox
)
from PySide6.QtCore import Qt, Signal
import re
import sys
from typing import List

_VALID_RE = re.compile(r"^[A-Za-z ]+$")  # letters and spaces only

class TextEntryWithCheckbox(QWidget):
    toggled = Signal()
    edited = Signal(str)
    delete_requested = Signal(object)

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.checkbox = QCheckBox(self)
        self.checkbox.setChecked(True)

        self.text_entry = QLineEdit(self)
        self.text_entry.setText(text)
        self.text_entry.setEnabled(True)

        self.status = QLabel(self)
        self.status.setFixedWidth(16)

        self.btn_delete = QPushButton("✖", self)
        self.btn_delete.setToolTip("Delete this class")
        self.btn_delete.setFixedWidth(28)

        self.checkbox.stateChanged.connect(self._on_toggled)
        self.text_entry.textChanged.connect(self._on_text_changed)
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self))

        layout.addWidget(self.checkbox)
        layout.addWidget(self.text_entry, 1)
        layout.addWidget(self.status)
        layout.addWidget(self.btn_delete)

    def _on_toggled(self, state: int):
        self.text_entry.setEnabled(state == Qt.Checked)
        self.toggled.emit()

    def _on_text_changed(self, txt: str):
        # Live validation: show hint if invalid characters are entered
        if not _VALID_RE.match(txt) and txt != "":
            self.text_entry.setStyleSheet("background-color: lightyellow;")
            self.text_entry.setToolTip("Only letters and spaces are allowed.")
        else:
            self.text_entry.setStyleSheet("")
            self.text_entry.setToolTip("")
        self.edited.emit(txt)

    def set_text(self, text: str):
        self.text_entry.setText(text)

    def get_text(self) -> str:
        return self.text_entry.text()

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_checked(self, v: bool):
        self.checkbox.setChecked(v)

    def set_delete_enabled(self, enabled: bool):
        self.btn_delete.setEnabled(enabled)

    def mark_invalid(self, reason: str):
        self.text_entry.setStyleSheet("background-color: lightpink;")
        self.text_entry.setToolTip(reason)
        self.status.setText("⚠️")

    def mark_changed(self):
        self.text_entry.setStyleSheet("background-color: lightblue;")
        self.text_entry.setToolTip("Changed from baseline")
        self.status.setText("✎")

    def clear_mark(self):
        self.text_entry.setStyleSheet("")
        self.text_entry.setToolTip("")
        self.status.setText("")

class ClassEntryWidget(QWidget):
    toggled = Signal(list)
    applied = Signal(list, list, list)

    def __init__(self, classes: List[str]):
        super().__init__()
        self._orig_classes = list(classes)
        self._classes = list(classes)
        self.entries: List[TextEntryWithCheckbox] = []
        self._can_be_applied = True
        self._build_ui()

    def disable_apply(self):
        self._can_be_applied = False
        self._update_button_state()

    def enable_apply(self):
        self._can_be_applied = True
        self._update_button_state()

    def get_classes(self) -> List[str]:
        return list(self._classes)

    def get_selected_classes(self) -> List[str]:
        return [e.get_text() for e in self.entries if e.is_checked()]

    def _build_ui(self):
        self.setWindowTitle("Class Entry Widget")
        self.v = QVBoxLayout(self)

        self.entries_container = QWidget(self)
        self.entries_layout = QVBoxLayout(self.entries_container)
        self.entries_layout.setContentsMargins(0, 0, 0, 0)
        self.entries_layout.setSpacing(6)
        self.v.addWidget(self.entries_container)

        for text in self._orig_classes:
            self._add_entry(text)

        self.v.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.btn_add = QPushButton("+ Add class", self)
        self.btn_add.clicked.connect(lambda: self._add_entry(focus=True))
        self.v.addWidget(self.btn_add)

        self.apply_button = QPushButton("Apply", self)
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply_changes)
        self.v.addWidget(self.apply_button)

        self._update_delete_buttons()
        self._update_button_state()

    def _add_entry(self, text: str | None = None, focus: bool = False):
        if text is None:
            text = ""
        ew = TextEntryWithCheckbox(text, self)
        ew.toggled.connect(self._on_entry_toggled)
        ew.edited.connect(lambda _txt: self._update_button_state())
        ew.delete_requested.connect(self._on_delete_entry)
        self.entries.append(ew)
        self.entries_layout.addWidget(ew)
        if focus:
            ew.text_entry.setFocus()
            ew.text_entry.end(False)
        self._update_delete_buttons()
        self._update_button_state()
        return ew

    def _on_delete_entry(self, entry: TextEntryWithCheckbox):
        if len(self.entries) <= 1:
            QMessageBox.information(self, "Cannot delete", "At least one class must remain.")
            return
        self.entries.remove(entry)
        entry.setParent(None)
        entry.deleteLater()
        self._update_delete_buttons()
        self._update_button_state()

    def _update_delete_buttons(self):
        allow_delete = len(self.entries) > 1
        for e in self.entries:
            e.set_delete_enabled(allow_delete)

    def _validate(self):
        texts = [e.get_text() for e in self.entries]
        lower_counts = {}
        for t in texts:
            lower_counts[t.lower()] = lower_counts.get(t.lower(), 0) + 1

        any_changes = len(texts) != len(self._orig_classes)
        valid_all = True
        for i, (e, text) in enumerate(zip(self.entries, texts)):
            e.clear_mark()
            if not text or not _VALID_RE.match(text):
                e.mark_invalid("Use letters and spaces only. No numbers or symbols.")
                valid_all = False
                continue
            if lower_counts[text.lower()] > 1:
                e.mark_invalid("Duplicate class name.")
                valid_all = False
                continue
            baseline = self._orig_classes[i] if i < len(self._orig_classes) else ""
            if text != baseline:
                any_changes = True
                e.mark_changed()
        return valid_all, any_changes, texts

    def _update_button_state(self):
        valid, changed, _ = self._validate()
        if hasattr(self, "apply_button"):
            self.apply_button.setEnabled(self._can_be_applied and valid and changed)

    def _on_entry_toggled(self):
        selected = [e.get_text() for e in self.entries if e.is_checked()]
        self.toggled.emit(selected)

    def _apply_changes(self):
        old = list(self._classes)
        valid, changed, texts = self._validate()
        if not (valid and changed):
            return
        self._classes = list(texts)
        self._orig_classes = list(texts)
        for e in self.entries:
            e.clear_mark()
        self.apply_button.setEnabled(False)
        self.applied.emit(old, list(self._classes), self.get_selected_classes())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    classes = ["person"]
    widget = ClassEntryWidget(classes)
    widget.applied.connect(lambda old, new, sel: print("Applied:\n old:", old, "\n new:", new, "\n selected:", sel))
    widget.resize(400, 250)
    widget.show()
    sys.exit(app.exec())
