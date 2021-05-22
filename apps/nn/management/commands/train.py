import os
from multiprocessing import cpu_count

from django.core.management import BaseCommand
from joblib import Parallel, delayed

from apps.nn.ImageHandler import ImageHandler
from apps.nn.NeuralNetwork import NeuralNetwork
from apps.nn.consts import IMAGES_ORIGINAL_PATH, IMAGES_COUNT, LEARNING_PART
from apps.nn.utils import get_logger, lead_time_writer

logger = get_logger(__name__)


class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        self.setup()
        logger.debug('load images')
        dataset_images = self.load_images()
        input_data = [x.l for x in dataset_images]
        output_data = [x.ab for x in dataset_images]
        logger.debug('init nn')
        nn = NeuralNetwork()
        logger.debug('train nn')
        nn.train(input_data, output_data)
        # predict_from_train(nn, dataset_images, 2)
        # self.predict_from_test(nn, dataset_images, 5)

    @staticmethod
    def setup():
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    @staticmethod
    def load_image(image_path):
        return ImageHandler(image_path).dataset_image

    @lead_time_writer
    def load_images(self):
        files = os.listdir(IMAGES_ORIGINAL_PATH)[:IMAGES_COUNT]
        results = Parallel(n_jobs=cpu_count())(delayed(self.load_image)(i) for i in files)
        return results

    @staticmethod
    def predict_image(nn, images):
        nn.predict_many(images)

        for image in images:
            image.show_original_gray_image()
            image.show_predicted_image()
            image.show_original_colored_image()

    def predict_from_train(self, nn, images, count=3):
        test_images = images[0:count]
        self.predict_image(nn, test_images)

    def predict_from_test(self, nn, images, count=3):
        test_images = images[int(IMAGES_COUNT * LEARNING_PART):int(IMAGES_COUNT * LEARNING_PART) + count]
        self.predict_image(nn, test_images)
