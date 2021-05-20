import os
from multiprocessing import cpu_count

import numpy as np
from joblib import Parallel, delayed

from ImageHandler import ImageHandler
from NeuralNetwork import NeuralNetwork
from consts import IMAGES_ORIGINAL_PATH, IMAGES_COUNT, LEARNING_PART, IMAGE_SIZE
from utils import lead_time_writer, get_logger
import tensorflow as tf
logger = get_logger(__name__)


def setup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_image(image_path):
    return ImageHandler(image_path).dataset_image


@lead_time_writer
def load_images():
    files = os.listdir(IMAGES_ORIGINAL_PATH)[:IMAGES_COUNT]
    results = Parallel(n_jobs=cpu_count())(delayed(load_image)(i) for i in files)
    return results


def predict_image(nn, images):
    nn.predict_many(images)

    for image in images:
        image.show_original_gray_image()
        image.show_predicted_image()
        image.show_original_colored_image()


def predict_from_train(nn, images, count=3):
    test_images = images[0:count]
    predict_image(nn, test_images)


def predict_from_test(nn, images, count=3):
    test_images = images[int(IMAGES_COUNT * LEARNING_PART):int(IMAGES_COUNT * LEARNING_PART) + count]
    predict_image(nn, test_images)


if __name__ == '__main__':
    setup()
    logger.debug('load images')
    dataset_images = load_images()
    input_data = [x.l for x in dataset_images]
    output_data = [x.ab for x in dataset_images]
    logger.debug('init nn')
    nn = NeuralNetwork()
    logger.debug('train nn')
    nn.train(input_data, output_data)
    # predict_from_train(nn, dataset_images, 2)
    predict_from_test(nn, dataset_images, 5)