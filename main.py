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
    test_images = images[0:count + 1]
    predict_image(nn, test_images)


def predict_from_test(nn, images, count=3):
    test_images = images[int(IMAGES_COUNT * LEARNING_PART):int(IMAGES_COUNT * LEARNING_PART) + count + 1]
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
    test_images = dataset_images[int(IMAGES_COUNT * LEARNING_PART) + 1:int(IMAGES_COUNT * LEARNING_PART) + 1+6]
    predict_from_train(nn, dataset_images, 2)
    predict_from_test(nn, dataset_images, 2)

    # test_image = dataset_images[0]
    # res = nn.predict(test_image)

    # y_pred = test_image.get_predicted_image()
    # y_real = test_image.get_original_colored_image()
    # ab_pred = test_image.ab
    # ab_real = test_image.predicted_ab
    # l = np.ones((64, 64, 1)) / 3
    # pred_lab = test_image.build_lab(l, ab_pred)
    # real_lab = test_image.build_lab(l, ab_real)
    # pred_rgb = test_image.lab2rgb(pred_lab)
    # real_rgb = test_image.lab2rgb(real_lab)
    # print(tf.reduce_mean(tf.image.ssim(pred_lab, real_lab, 1.0)))
    # print(tf.reduce_mean(tf.image.ssim(pred_rgb, real_rgb, 1.0)))

