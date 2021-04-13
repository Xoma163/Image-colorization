import os

from joblib import Parallel, delayed

from ImageHandler import ImageHandler
from NeuralNetwork import NeuralNetwork
from consts import IMAGES_ORIGINAL_PATH, IMAGES_COUNT, LEARNING_PART, IMAGES_RESIZED_PATH
from utils import lead_time_writer
from multiprocessing import cpu_count

def setup():
    os.makedirs(IMAGES_RESIZED_PATH, exist_ok=True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_image(image_path):
    return ImageHandler(image_path).dataset_image


@lead_time_writer
def load_images():
    files = os.listdir(IMAGES_ORIGINAL_PATH)[:IMAGES_COUNT]
    results = Parallel(n_jobs=cpu_count())(delayed(load_image)(i) for i in files)
    return results


def predict_image(images):
    nn.predict_many(images)

    for image in images:
        image.show_original_gray_image()
        image.show_predicted_image()
        image.show_original_colored_image()


def predict_from_train(images, count=3):
    test_images = images[0:count + 1]
    predict_image(test_images)


def predict_from_test(images, count=3):
    test_images = images[int(IMAGES_COUNT * LEARNING_PART):int(IMAGES_COUNT * LEARNING_PART) + count + 1]
    predict_image(test_images)


if __name__ == '__main__':
    setup()

    print('load images')
    dataset_images = load_images()
    input_data = [x.l for x in dataset_images]
    output_data = [x.ab for x in dataset_images]

    print('init NN')
    nn = NeuralNetwork()
    print('train NN')
    nn.train(input_data, output_data)
    nn.show_loss_graphic()

    predict_from_train(dataset_images, 3)
    predict_from_test(dataset_images, 3)
