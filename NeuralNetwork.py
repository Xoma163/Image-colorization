import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

from consts import LEARNING_PART, IMAGE_SIZE, EPOCHS


class NeuralNetwork:
    """
    Свёрточная нейронная сеть
    """
    def __init__(self):
        self.model = models.Sequential([
            layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(3, (3, 3), activation='relu', padding='same'),
        ])

    def train(self, input_data, output_data):
        """
        Обучение модели
        """
        index = int(len(input_data) * LEARNING_PART)

        data_train_x = np.array(input_data[:index])
        data_test_x = np.array(input_data[index:])

        data_train_y = np.array(output_data[:index])
        data_test_y = np.array(output_data[index:])

        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(x=data_train_x, y=data_train_y, epochs=EPOCHS, batch_size=32, shuffle=True)

        # test_accuracy = self.model.evaluate(x=data_test_x, y=data_test_y)
        # print('Точность: ', test_accuracy)


        test_x_image = data_test_x[1].reshape((1,) + (data_test_x[1].shape))
        predicted_image = self.model.predict(test_x_image)
        predicted_image = predicted_image.reshape(predicted_image.shape[1:]).astype(np.uint8)
        img = Image.fromarray(predicted_image)
        plt.imshow(img)
        plt.show()

        test_x_image = data_test_x[1].astype(np.uint8)
        test_x_image = test_x_image.reshape(test_x_image.shape[:2])
        img = Image.fromarray(test_x_image).convert('RGB')
        plt.imshow(img)
        plt.show()

        test_y_image = data_test_y[1].astype(np.uint8)
        img = Image.fromarray(test_y_image)
        plt.imshow(img)
        plt.show()