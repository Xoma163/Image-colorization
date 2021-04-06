import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

from consts import IMAGES_ORIGINAL_PATH, IMAGE_SIZE, IMAGES_RESIZED_PATH


class DatasetImage:
    MAX_L = 100
    MAX_AB = 128
    MAX_RGB = 255

    def __init__(self, rgb):
        self.rgb = rgb  # denormalized
        lab = self.rgb2lab(self.rgb)  # denormalized

        self.l = self._normalize_l(self._get_l(lab))  # normalized
        self.ab = self._normalize_ab(self._get_ab(lab))  # normalized
        self.predicted_ab = None  # normalized

    def set_predicted_ab(self, ab):
        self.predicted_ab = ab

    @staticmethod
    def _get_l(lab):
        return lab[:, :, 0:1]

    @staticmethod
    def _get_ab(lab):
        return lab[:, :, 1:]

    @staticmethod
    def rgb2lab(rgb):
        return rgb2lab(rgb)

    def lab2rgb(self, lab):
        rgb = lab2rgb(lab) * self.MAX_RGB
        rgb = rgb.astype(np.uint8)
        return rgb

    def show_original_gray_image(self):
        lab = self.build_lab(self._denormalize_l(self.l), np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2)))
        rgb = self.lab2rgb(lab)
        self.show_image(rgb)

    def show_predicted_image(self):
        lab = self.build_lab(self._denormalize_l(self.l), self._denormalize_ab(self.predicted_ab))
        rgb = self.lab2rgb(lab)
        self.show_image(rgb)

    def show_original_colored_image(self):
        self.show_image(self.rgb)

    @staticmethod
    def show_image(rgb):
        image = Image.fromarray(rgb)
        plt.imshow(image)
        plt.show()

    @staticmethod
    def build_lab(l, ab):
        lab = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        lab[:, :, 0] = l[:, :, 0]
        lab[:, :, 1:] = ab
        return lab

    def _normalize_l(self, l):
        return l / self.MAX_L

    def _normalize_ab(self, ab):
        return ab / self.MAX_AB

    def _denormalize_l(self, l):
        return l * self.MAX_L

    def _denormalize_ab(self, ab):
        return ab * self.MAX_AB


class ImageHandler:

    def __init__(self, filename: str):
        """
        Инициализация обработчика изображения
        При запуске автоматически нарезает и сохраняет картинки для датасета
        """
        self.filename = filename
        self.dataset_image = None
        self.set_dataset_image()

    def set_dataset_image(self):
        """
        Подготавливает изображения для датасета нейронной сети
        Генерирует два изображения с измененным размером - цветную и чёрно-белую
        """

        original_image_path = os.path.join(IMAGES_ORIGINAL_PATH, self.filename)
        resized_image_path = os.path.join(IMAGES_RESIZED_PATH, self.filename)

        original_image = Image.open(original_image_path)

        if not os.path.exists(resized_image_path):
            resized_image = original_image.resize((IMAGE_SIZE, IMAGE_SIZE))
            resized_image.save(resized_image_path)
            image_color = resized_image
        else:
            image_color = Image.open(resized_image_path)
        self.dataset_image = DatasetImage(np.array(image_color))
        original_image.close()
        image_color.close()
