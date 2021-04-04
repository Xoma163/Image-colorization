import numpy as np
from PIL import Image
from tensorflow.keras import optimizers, models, layers

from ImageHandler import ImageHandler, show_image
from consts import LEARNING_PART, EPOCHS, IMAGE_SIZE


class NeuralNetwork:
    """
    Свёрточная нейронная сеть
    """

    def __init__(self):
        self.model = models.Sequential([
            layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            # layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            # layers.MaxPool2D(2),
            # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            # layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            # layers.UpSampling2D(2),
            # layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(3, (3, 3), activation='relu', padding='same'),
        ])

        self.model.summary()
        self.model.compile(optimizer=optimizers.Adam(), loss='mse')

    def train(self, input_data, output_data):
        """
        Обучение модели
        """
        slicer_index = int(len(input_data) * LEARNING_PART)

        data_train_x = np.array(input_data[:slicer_index])
        data_test_x = np.array(input_data[slicer_index:])

        data_train_y = np.array(output_data[:slicer_index])
        data_test_y = np.array(output_data[slicer_index:])

        self.model.fit(
            x=data_train_x,
            y=data_train_y,
            epochs=EPOCHS,
            batch_size=16,
            shuffle=True
        )

        test_accuracy = self.model.evaluate(x=data_test_x, y=data_test_y)
        print('Точность: ', test_accuracy)

    def pretict_from_path(self, image_path):
        pass

        # test_x_image = Image.open(image_path)
        # test_x_image = np.array(test_x_image) / 255
        # test_x_image = test_x_image.reshape((1,) + test_x_image.shape)

    def predict_from_dataset(self, image: ImageHandler):
        bw_matrix = image.bw_matrix
        bw_matrix = bw_matrix.reshape((1,) + bw_matrix.shape)
        predicted_image = self.model.predict(bw_matrix)

        predicted_image = predicted_image.reshape(predicted_image.shape[1:]) * 255
        predicted_image = predicted_image.astype(np.uint8)
        img = Image.fromarray(predicted_image)

        show_image(img)
        image.show_color_image()
