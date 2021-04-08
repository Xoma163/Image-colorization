import os

from ImageHandler import ImageHandler
from NeuralNetwork import NeuralNetwork
from consts import IMAGES_ORIGINAL_PATH, IMAGES_COUNT, LEARNING_PART, IMAGES_RESIZED_PATH
from utils import CyclePercentWriter, lead_time_writer


@lead_time_writer
def load_images():
    dataset_images = []
    cpw = CyclePercentWriter(IMAGES_COUNT, per=20)
    for i, image_path in enumerate(os.listdir(IMAGES_ORIGINAL_PATH)[:IMAGES_COUNT]):
        dataset_images.append(ImageHandler(image_path).dataset_image)
        if cpw.check(i):
            print(f"{i}/{IMAGES_COUNT}")
    return dataset_images
    # return [ImageHandler(image_path).dataset_image for image_path in os.listdir(IMAGES_ORIGINAL_PATH)[:IMAGES_COUNT]]


def setup():
    os.makedirs(IMAGES_RESIZED_PATH, exist_ok=True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    setup()

    print('load images')
    images = load_images()
    input_data = [x.l for x in images]
    output_data = [x.ab for x in images]

    print('init NN')
    nn = NeuralNetwork()
    print('train NN')
    nn.train(input_data, output_data)
    nn.show_loss_graphic()

    # test_images = images[0:2] + images[int(IMAGES_COUNT * LEARNING_PART):int(IMAGES_COUNT * LEARNING_PART) + 2]
    # nn.predict_many(test_images)
    #
    # for image in test_images:
    #     # image.show_original_gray_image()
    #     image.show_predicted_image()
    #     image.show_original_colored_image()
