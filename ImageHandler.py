import os

import numpy as np
from PIL import Image

from consts import IMAGES_ORIGINAL_PATH, IMAGES_RESIZED_GRAY_PATH, IMAGES_RESIZED_COLORED_PATH, IMAGE_SIZE


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

        self.image_color = None
        self.image_gray = None

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
            self.image_color = resized_image
        else:
            self.image_color = Image.open(self.resized_colored_image_path)
        self.image_color.load()

        if not os.path.exists(self.resized_gray_image_path):
            bw_image = self.image_color.convert('L')  # конвертация изображения в серые цвета
            bw_image.save(self.resized_gray_image_path)
            self.image_gray = bw_image
        else:
            self.image_gray = Image.open(self.resized_colored_image_path)
            self.image_gray = self.image_gray.convert('L')  # при открытии файла он пытается открыть его как RGB
        self.image_gray.load()

    # def rgb2gray(self, rgb):
    #     return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

    def get_color_as_matrix(self):
        """
        Перевод цветной картинки в матрицу, где ячейки это [R, G, B]
        Итоговый shape: (IMAGE_SIZE, IMAGE_SIZE, 3)
        """
        return self._get_as_matrix(self.image_color)

    def get_gray_as_matrix(self):
        """
        Перевод ЧБ картинки в матрицу, где ячейки это [Gray]
        Итоговый shape: (IMAGE_SIZE, IMAGE_SIZE, 1)
        """
        matrix = self._get_as_matrix(self.image_gray)
        return matrix.reshape(matrix.shape + (1,))

    def _get_as_matrix(self, image):
        return np.array(image)
