from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

NOISE_LIST_PATTERN = re.compile(
    r"^\s*\[(?P<components>(?:\d+\s*(?:,\s*\d+\s*)*)?)\]\s*$"
)


def read_noise_components(noise_list_file: str | Path) -> set[int]:
    noise_list_file = Path(noise_list_file)
    if not noise_list_file.is_file():
        return set()

    for line in noise_list_file.read_text(encoding="utf-8").splitlines():
        match = NOISE_LIST_PATTERN.fullmatch(line)
        if match is not None:
            components = match.group("components")
            return (
                {int(component.strip()) for component in components.split(",")}
                if components
                else set()
            )

    return set()


def write_noise_components(
    noise_list_file: str | Path,
    components: set[int],
) -> None:
    noise_list_file = Path(noise_list_file)
    lines = (
        noise_list_file.read_text(encoding="utf-8").splitlines()
        if noise_list_file.is_file()
        else []
    )
    component_line = f"[{', '.join(map(str, sorted(components)))}]"

    for index, line in enumerate(lines):
        if NOISE_LIST_PATTERN.fullmatch(line) is not None:
            lines[index] = component_line
            break
    else:
        lines.append(component_line)

    noise_list_file.parent.mkdir(parents=True, exist_ok=True)
    noise_list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ComponentLabeler(QMainWindow):
    def __init__(self, scans: Mapping[str, Mapping[str, str | int]]):
        super().__init__()
        if not scans:
            raise ValueError("No scans are available for component labelling.")

        self.scans = dict(scans)
        self.scan_names = list(self.scans)
        self.scan_index = 0
        self.ic = 1

        self.setWindowTitle("FixRABIES component labelling")
        self._build_ui()
        self._set_dark_theme()
        self._refresh()

    @property
    def current_scan_name(self) -> str:
        return self.scan_names[self.scan_index]

    @property
    def current_scan(self) -> Mapping[str, str | int]:
        return self.scans[self.current_scan_name]

    @property
    def current_noise_file(self) -> Path:
        return Path(str(self.current_scan["noise_list"]))

    @property
    def current_n_ic(self) -> int:
        return int(self.current_scan["n_ic"])

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        scan_layout = QHBoxLayout()
        scan_layout.addWidget(QLabel("Scan:"))
        self.scan_combo = QComboBox()
        self.scan_combo.addItems(self.scan_names)
        self.scan_combo.currentIndexChanged.connect(self._select_scan)
        scan_layout.addWidget(self.scan_combo)
        layout.addLayout(scan_layout)

        self.component_status = QLineEdit()
        self.component_status.setReadOnly(True)
        layout.addWidget(self.component_status)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        self.image_labels: list[QLabel] = []
        for _ in range(3):
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(500, 300)
            image_label.setStyleSheet("border: 1px solid #555;")
            self.image_labels.append(image_label)
            image_layout.addWidget(image_label)
        scroll_area.setWidget(image_container)
        layout.addWidget(scroll_area)

        rating_layout = QHBoxLayout()
        for label, rating in (
            ("Noise", "noise"),
            ("Unsure", "unsure"),
            ("Signal", "signal"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda checked=False, value=rating: self._record_rating(value)
            )
            rating_layout.addWidget(button)
        layout.addLayout(rating_layout)

        navigation_layout = QHBoxLayout()
        back_button = QPushButton("Back")
        back_button.clicked.connect(self._go_back)
        forward_button = QPushButton("Forward")
        forward_button.clicked.connect(self._go_forward)
        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(self.close)
        navigation_layout.addWidget(back_button)
        navigation_layout.addWidget(forward_button)
        navigation_layout.addStretch()
        navigation_layout.addWidget(quit_button)
        layout.addLayout(navigation_layout)

    def _set_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #202124; color: #e8eaed; }
            QLineEdit, QComboBox {
                background-color: #303134; border: 1px solid #5f6368;
                border-radius: 4px; padding: 6px; color: #e8eaed;
            }
            QPushButton {
                background-color: #3c4043; border: 1px solid #5f6368;
                border-radius: 4px; padding: 7px 14px; color: #e8eaed;
            }
            QPushButton:hover { background-color: #4a4e52; }
            """
        )

    def _select_scan(self, index: int) -> None:
        self.scan_index = index
        self.ic = 1
        self._refresh()

    def _set_position(self, scan_index: int, ic: int) -> None:
        self.scan_index = scan_index
        self.ic = ic
        self.scan_combo.blockSignals(True)
        self.scan_combo.setCurrentIndex(scan_index)
        self.scan_combo.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        noise_components = read_noise_components(self.current_noise_file)
        status = "noise" if self.ic in noise_components else "not-noise"
        self.component_status.setText(
            f"{self.current_scan_name} - IC {self.ic}/{self.current_n_ic}: {status}"
        )
        report_dir = (
            Path(str(self.current_scan["scan_path"]))
            / "filtered_func_data.ica"
            / "report"
        )
        image_names = [
            f"IC_{self.ic}_thresh.png",
            f"t{self.ic}.png",
            f"f{self.ic}.png",
        ]
        for image_label, image_name in zip(
            self.image_labels,
            image_names,
            strict=True,
        ):
            image_path = report_dir / image_name
            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                image_label.setPixmap(QPixmap())
                image_label.setText(f"Missing:\n{image_path}")
                continue
            image_label.setPixmap(
                pixmap.scaled(
                    700,
                    500,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            image_label.setText("")

    def _record_rating(self, rating: str) -> None:
        components = read_noise_components(self.current_noise_file)
        if rating == "noise":
            components.add(self.ic)
        else:
            components.discard(self.ic)
        write_noise_components(self.current_noise_file, components)
        self._go_forward()

    def _go_back(self) -> None:
        if self.ic > 1:
            self._set_position(self.scan_index, self.ic - 1)
        elif self.scan_index > 0:
            previous_scan_index = self.scan_index - 1
            previous_n_ic = int(
                self.scans[self.scan_names[previous_scan_index]]["n_ic"]
            )
            self._set_position(previous_scan_index, previous_n_ic)

    def _go_forward(self) -> None:
        if self.ic < self.current_n_ic:
            self._set_position(self.scan_index, self.ic + 1)
        elif self.scan_index < len(self.scan_names) - 1:
            self._set_position(self.scan_index + 1, 1)


def launch_component_labeler(scans: Mapping[str, Mapping[str, str | int]]) -> None:
    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(sys.argv)

    window = ComponentLabeler(scans)
    window.resize(900, 1000)
    window.show()

    if owns_application:
        application.exec()
