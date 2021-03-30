import os

from ImageHandler import ImageHandler
from NeuralNetwork import NeuralNetwork
from consts import IMAGES_ORIGINAL_PATH, IMAGES_RESIZED_COLORED_PATH, IMAGES_RESIZED_GRAY_PATH, IMAGES_COUNT


def load_images():
    return [ImageHandler(image) for image in os.listdir(IMAGES_ORIGINAL_PATH)[:IMAGES_COUNT]]


def setup():
    os.makedirs(IMAGES_RESIZED_COLORED_PATH, exist_ok=True)
    os.makedirs(IMAGES_RESIZED_GRAY_PATH, exist_ok=True)


if __name__ == '__main__':
    setup()

    dataset = load_images()
    data_train = [x.get_gray_as_matrix() for x in dataset]
    data_test = [x.get_color_as_matrix() for x in dataset]
    nn = NeuralNetwork()
    nn.train(data_train, data_test)
