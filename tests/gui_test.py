# viewer.py

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

IMAGE_NAMES = ["image1.png", "image2.png", "image3.png"]


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Rating")

        self.subjects = ["sub-1", "sub-2", "sub-3"]

        self.subject_idx = 0
        self.visit_idx = 1

        self._build_ui()
        self.load_images()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        # ------------------------------------------------------------------
        # Subject selection
        # ------------------------------------------------------------------
        subject_layout = QHBoxLayout()

        subject_label = QLabel("Subject:")
        self.subject_combo = QComboBox()
        self.subject_combo.addItems(self.subjects)
        self.subject_combo.currentIndexChanged.connect(self.on_subject_changed)

        subject_layout.addWidget(subject_label)
        subject_layout.addWidget(self.subject_combo)
        subject_layout.addStretch()

        layout.addLayout(subject_layout)

        # ------------------------------------------------------------------
        # Images
        # ------------------------------------------------------------------
        image_layout = QHBoxLayout()

        self.image_labels = []

        for _ in range(3):
            label = QLabel("No image")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(300, 300)
            label.setStyleSheet("border: 1px solid gray; background:white;")

            self.image_labels.append(label)
            image_layout.addWidget(label)

        layout.addLayout(image_layout)

        # ------------------------------------------------------------------
        # Rating buttons
        # ------------------------------------------------------------------
        rating_layout = QHBoxLayout()

        self.noise_btn = QPushButton("Noise")
        self.unsure_btn = QPushButton("Unsure")
        self.signal_btn = QPushButton("Signal")

        self.noise_btn.clicked.connect(lambda: self.record_rating("noise"))
        self.unsure_btn.clicked.connect(lambda: self.record_rating("unsure"))
        self.signal_btn.clicked.connect(lambda: self.record_rating("signal"))

        rating_layout.addWidget(self.noise_btn)
        rating_layout.addWidget(self.unsure_btn)
        rating_layout.addWidget(self.signal_btn)

        layout.addLayout(rating_layout)

        # ------------------------------------------------------------------
        # Navigation
        # ------------------------------------------------------------------
        nav_layout = QHBoxLayout()

        self.back_btn = QPushButton("Back")
        self.forward_btn = QPushButton("Forward")
        self.quit_btn = QPushButton("Quit")

        self.back_btn.clicked.connect(self.go_back)
        self.forward_btn.clicked.connect(self.go_forward)
        self.quit_btn.clicked.connect(self.close)

        nav_layout.addWidget(self.back_btn)
        nav_layout.addWidget(self.forward_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.quit_btn)

        layout.addLayout(nav_layout)

        # Status label
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

    def current_folder(self):
        """
        Example:
        sub-1/1/
        sub-1/2/
        sub-2/1/
        """
        subject = self.subjects[self.subject_idx]
        return Path(subject) / str(self.visit_idx)

    def load_images(self):
        folder = self.current_folder()

        for label, image_name in zip(
            self.image_labels,
            IMAGE_NAMES,
            strict=False,
        ):
            image_path = folder / image_name

            if image_path.exists():
                pixmap = QPixmap(str(image_path))

                pixmap = pixmap.scaled(
                    400,
                    400,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )

                label.setPixmap(pixmap)
                label.setText("")
            else:
                label.setPixmap(QPixmap())
                label.setText(f"Missing:\n{image_path}")

        self.status_label.setText(
            f"Subject: {self.subjects[self.subject_idx]}   " f"Index: {self.visit_idx}"
        )

    def on_subject_changed(self, idx):
        self.subject_idx = idx
        self.visit_idx = 1
        self.load_images()

    def record_rating(self, rating):
        subject = self.subjects[self.subject_idx]

        print(f"subject={subject}, " f"index={self.visit_idx}, " f"rating={rating}")

    def go_back(self):
        if self.visit_idx > 1:
            self.visit_idx -= 1
            self.load_images()

    def go_forward(self):
        self.visit_idx += 1
        self.load_images()


if __name__ == "__main__":
    app = QApplication([])

    window = ImageViewer()
    window.resize(1400, 700)
    window.show()

    app.exec()
