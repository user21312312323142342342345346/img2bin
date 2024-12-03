import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QPushButton, QLabel, QFileDialog, QSlider, QLineEdit, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.canvas_size = (96, 22)  # 96x22 cells for a 960x220 image
        self.cell_size = 10  # Each cell is 10x10 pixels
        self.grid = np.zeros((self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)  # Store binary values (0-1)
        self.image = None  # Placeholder for the loaded image
        self.opacity = 0.5  # Default opacity for overlay
        self.block_size = 15  # Default block size
        self.constant = 10  # Default constant
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Draw on Image (960x220 version)')

        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.canvas_size[0] * self.cell_size, self.canvas_size[1] * self.cell_size)
        layout.addWidget(self.image_label)

        # Filename label
        self.filename_label = QLabel("No image loaded", self)
        layout.addWidget(self.filename_label)

        # Button to load image
        self.load_button = QPushButton('Open Image', self)
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        # Block Size Slider
        block_size_layout = QHBoxLayout()
        self.block_size_label = QLabel(f'Block Size: {self.block_size}', self)
        block_size_layout.addWidget(self.block_size_label)

        self.block_size_slider = QSlider(Qt.Horizontal, self)
        self.block_size_slider.setRange(3, 63)  # Ensure the range is valid for adaptive threshold
        self.block_size_slider.setValue(self.block_size)
        self.block_size_slider.valueChanged.connect(self.update_block_size)
        block_size_layout.addWidget(self.block_size_slider)

        layout.addLayout(block_size_layout)

        # Constant Slider
        constant_layout = QHBoxLayout()
        self.constant_label = QLabel(f'Constant: {self.constant}', self)
        constant_layout.addWidget(self.constant_label)

        self.constant_slider = QSlider(Qt.Horizontal, self)
        self.constant_slider.setRange(0, 63)
        self.constant_slider.setValue(self.constant)
        self.constant_slider.valueChanged.connect(self.update_constant)
        constant_layout.addWidget(self.constant_slider)

        layout.addLayout(constant_layout)

        # Opacity Slider
        opacity_layout = QHBoxLayout()
        self.opacity_label = QLabel(f'Opacity: {self.opacity:.2f}', self)
        opacity_layout.addWidget(self.opacity_label)

        self.opacity_slider = QSlider(Qt.Horizontal, self)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.opacity * 100))
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        opacity_layout.addWidget(self.opacity_slider)

        layout.addLayout(opacity_layout)

        # Clear Canvas Button
        self.clear_button = QPushButton('Clear Canvas')  # Clear button
        self.clear_button.clicked.connect(self.clear_canvas)  # Connect to clear method
        layout.addWidget(self.clear_button)

        # Hex code display
        self.hex_output = QLineEdit(self)
        self.hex_output.setPlaceholderText('Hex code will appear here')
        layout.addWidget(self.hex_output)

        # Button to export hex and copy to clipboard
        button_layout = QHBoxLayout()
        self.export_button = QPushButton('Export to Hex', self)
        self.export_button.clicked.connect(self.export_hex)
        button_layout.addWidget(self.export_button)

        self.copy_button = QPushButton('Copy Hex', self)
        self.copy_button.clicked.connect(self.copy_hex)
        button_layout.addWidget(self.copy_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.update_canvas()  # Display the initial blank canvas

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp *.jpeg)", options=options)
        if file_name:
            self.image = cv2.imread(file_name)
            # Resize image to fit canvas with 96x31 cells
            self.image = cv2.resize(self.image, (self.canvas_size[0] * self.cell_size, self.canvas_size[1] * self.cell_size))
            self.show_image()
            self.initialize_grid_from_image()  # Initialize the grid based on the loaded image
            self.filename_label.setText(f"Loaded Image: {file_name}")  # Update the filename label

    def show_image(self):
        if self.image is not None:
            # Convert BGR to RGB for PyQt5 display
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            height, width, channel = self.image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def initialize_grid_from_image(self):
        # Convert the loaded image to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        # Apply adaptive thresholding using the current parameters
        block_size = self.block_size_slider.value() | 1  # Ensure block size is odd
        constant = self.constant_slider.value()
        adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY, block_size, constant)

        # Update the grid based on the binary image
        for i in range(self.canvas_size[1]):
            for j in range(self.canvas_size[0]):  # Use the full width of the grid
                if adaptive_threshold[i * self.cell_size + self.cell_size // 2, j * self.cell_size + self.cell_size // 2] == 0:
                    self.grid[i, j] = 1  # Black
                else:
                    self.grid[i, j] = 0  # White

        self.update_canvas()  # Update the canvas to show the loaded image

    def update_canvas(self):
        # Create an empty image (overlay)
        overlay = np.ones((self.canvas_size[1] * self.cell_size, self.canvas_size[0] * self.cell_size, 3), dtype=np.uint8) * 255

        # Draw the cells based on grid data (black for 1, white for 0)
        for i in range(self.canvas_size[1]):
            for j in range(self.canvas_size[0]):
                if self.grid[i, j] == 1:
                    cv2.rectangle(overlay, (j * self.cell_size, i * self.cell_size),
                                  ((j + 1) * self.cell_size, (i + 1) * self.cell_size), (0, 0, 0), -1)

        # Blend the overlay with the image if an image is loaded
        if self.image is not None:
            image = cv2.addWeighted(overlay, self.opacity, self.image, 1 - self.opacity, 0)
        else:
            image = overlay

        # Update the displayed image
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def update_block_size(self):
        self.block_size = self.block_size_slider.value() | 1  # Ensure block size is odd
        self.block_size_label.setText(f'Block Size: {self.block_size}')  # Update label
        if self.image is not None:
            self.initialize_grid_from_image()

    def update_constant(self):
        self.constant = self.constant_slider.value()
        self.constant_label.setText(f'Constant: {self.constant}')  # Update label
        if self.image is not None:
            self.initialize_grid_from_image()

    def update_opacity(self):
        self.opacity = self.opacity_slider.value() / 100  # Convert to 0.0 - 1.0
        self.opacity_label.setText(f'Opacity: {self.opacity:.2f}')  # Update label
        self.update_canvas()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.flip_pixel(pos)

    def flip_pixel(self, pos):
        x = (pos.x() - self.image_label.x()) // self.cell_size
        y = (pos.y() - self.image_label.y()) // self.cell_size
        if 0 <= x < self.canvas_size[0] and 0 <= y < self.canvas_size[1]:
            self.grid[y, x] = 1 - self.grid[y, x]  # Flip the pixel value (black <-> white)
            self.update_canvas()

    def clear_canvas(self):
        self.grid.fill(0)  # Set all values in the grid to 0
        self.update_canvas()  # Update canvas to show cleared state

    def export_hex(self):
        flattened_grid = self.grid.flatten()  # Flatten the grid
        binary_string = ''.join(map(str, flattened_grid))

        # Group the binary string into 8-bit segments and convert to hex
        hex_values = [f'{int(binary_string[i:i + 8], 2):02X}' for i in range(0, len(binary_string), 8)]

        # Join the hex values into a string
        hex_output = ' '.join(hex_values)
        self.hex_output.setText(hex_output)

    def copy_hex(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.hex_output.text())
        QMessageBox.information(self, "Copied", "Hex code copied to clipboard!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageCanvas()
    window.show()
    sys.exit(app.exec_())
