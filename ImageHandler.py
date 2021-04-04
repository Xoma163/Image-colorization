import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from consts import IMAGES_ORIGINAL_PATH, IMAGES_RESIZED_COLORED_PATH, IMAGE_SIZE, IMAGES_RESIZED_GRAY_PATH


class ImageHandler:

    def __init__(self, filename: str):
        """
        Инициализация обработчика изображения
        При запуске автоматически нарезает и сохраняет картинки для датасета
        """
        self.filename = filename

        self.original_image_path = os.path.join(IMAGES_ORIGINAL_PATH, filename)
        self.resized_colored_image_path = os.path.join(IMAGES_RESIZED_COLORED_PATH, self.filename)
        self.resized_gray_image_path = os.path.join(IMAGES_RESIZED_GRAY_PATH, self.filename)

        self.original_image = Image.open(self.original_image_path)

        self.color_matrix = None
        self.bw_matrix = None

        self.prepare_dataset()
        self.original_image.close()

    def prepare_dataset(self):
        """
        Подготавливает изображения для датасета нейронной сети
        Генерирует два изображения с измененным размером - цветную и чёрно-белую
        """

        if not os.path.exists(self.resized_colored_image_path):
            resized_image = self.original_image.resize((IMAGE_SIZE, IMAGE_SIZE))
            resized_image.save(self.resized_colored_image_path)
            image_color = resized_image
        else:
            image_color = Image.open(self.resized_colored_image_path)

        if not os.path.exists(self.resized_gray_image_path):
            image_gray = image_color.convert('L')  # конвертация изображения в серые цвета
            image_gray.save(self.resized_gray_image_path)
        else:
            image_gray = Image.open(self.resized_gray_image_path)

        self.color_matrix = self.get_color_as_matrix(image_color)
        self.bw_matrix = self.get_gray_as_matrix(image_gray)

        image_color.close()
        image_gray.close()

    def get_color_as_matrix(self, image_color):
        """
        Перевод цветной картинки в матрицу, где ячейки это [R, G, B]
        Итоговый shape: (IMAGE_SIZE, IMAGE_SIZE, 3)
        """
        return self._get_as_matrix(image_color)

    # @staticmethod
    # def rgb2gray(rgb):
    #     return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def get_gray_as_matrix(self, image_gray):
        """
        Перевод ЧБ картинки в матрицу, где ячейки это [G]
        Итоговый shape: (IMAGE_SIZE, IMAGE_SIZE, 1)
        """
        matrix = self._get_as_matrix(image_gray)
        return matrix.reshape(matrix.shape + (1,))

    def _get_as_matrix(self, image):
        return np.array(image) / 255

    def show_color_image(self):
        self._show_image(self.color_matrix)

    def show_gray_image(self):
        self._show_image(self.bw_matrix)

    def _show_image(self, image):
        image = image * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)

        show_image(image)


def show_image(image):
    plt.imshow(image)
    plt.show()
