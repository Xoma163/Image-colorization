import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

from apps.nn.consts import IMAGE_SIZE, IMAGES_ORIGINAL_PATH, IMAGES_RESIZED_PATH


class DatasetImage:
    MAX_L = 100
    MAX_AB = 128
    MAX_RGB = 255

    def __init__(self, rgb=None, lab=None):
        self.l = None
        self.ab = None
        self.predicted_ab = None  # normalized
        if rgb is not None:
            lab = self.rgb2lab(rgb)  # denormalized
        self.l = self._normalize_l(self._get_l(lab))  # normalized
        self.ab = self._normalize_ab(self._get_ab(lab))  # normalized

    @property
    def lab(self):
        return self.build_lab(self._denormalize_l(self.l), self._denormalize_ab(self.ab))

    @property
    def a(self):
        return self.ab[:, :, :1]

    @property
    def b(self):
        return self.ab[:, :, 1:]

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

    def get_original_gray_image(self):
        return self.build_lab(self._denormalize_l(self.l), np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2)))

    def show_original_gray_image(self):
        lab = self.get_original_gray_image()
        rgb = self.lab2rgb(lab)
        self.show_image(rgb)

    def get_predicted_image(self):
        return self.build_lab(self._denormalize_l(self.l), self._denormalize_ab(self.predicted_ab))

    def show_predicted_image(self):
        lab = self.get_predicted_image()
        rgb = self.lab2rgb(lab)
        self.show_image(rgb)

    def get_original_colored_image(self):
        return self.build_lab(self._denormalize_l(self.l), self._denormalize_ab(self.ab))

    def show_original_colored_image(self):
        lab = self.get_original_colored_image()
        rgb = self.lab2rgb(lab)
        self.show_image(rgb)

    @staticmethod
    def show_image(rgb):
        image = Image.fromarray(rgb)
        plt.imshow(image)
        plt.show()

    @staticmethod
    def build_lab(l, ab):
        lab = np.zeros((l.shape[0], l.shape[1], 3))
        lab[:, :, 0] = l[:, :, 0]
        lab[:, :, 1:] = ab
        return lab

    def _normalize_l(self, l):
        return l / self.MAX_L

    def _normalize_ab(self, ab):
        # return ab / self.MAX_AB
        return ab / self.MAX_AB / 2 + 0.5

    def _denormalize_l(self, l):
        return l * self.MAX_L

    def _denormalize_ab(self, ab):
        # return ab * self.MAX_AB
        return (ab - 0.5) * 2 * self.MAX_AB


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
