import os
import time

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import optimizers, layers, models
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from ImageHandler import DatasetImage
from consts import LEARNING_PART, EPOCHS, IMAGE_SIZE
from utils import CyclePercentWriter


class LossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.loss = {}
        self.cpw = CyclePercentWriter(EPOCHS, per=5)

    def on_epoch_end(self, epoch, logs=None):
        self.loss[epoch] = (logs['loss'])
        if self.cpw.check(epoch):
            print(f"Epoch {epoch}/{EPOCHS}. Loss {round(logs['loss'], 5)}")


class NeuralNetwork:
    """
    Свёрточная нейронная сеть
    """
    WEIGHTS_FILE = 'model/model_weight'
    # https://github.com/keras-team/keras/issues/8123
    GPUS_COUNT = int(os.getenv("GPUS_COUNT"))

    def __init__(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPool2D(2),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPool2D(2),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.MaxPool2D(2),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D(2),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D(2),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(2, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D(2),
        ])
        if self.GPUS_COUNT == 1:
            self.model = model
        else:
            self.model = multi_gpu_model(model, gpus=self.GPUS_COUNT)
        self.model.summary()
        self.model.compile(optimizer=optimizers.Adam(), loss='mse')

        self.loss_callback = LossCallback()

    # def get_fit_data(self, data_train_x, data_train_y, batch_size=16):
    #     # [data_train_x[i:i + batch_size] for i in range(0, len(data_train_x), batch_size)]\
    #     data = [(data_train_x[i:i + batch_size],data_train_y[i:i + batch_size]) for i in range(len(data_train_x), batch_size)]
    #     for x, y in data:
    #         yield x, y

    def train(self, input_data, output_data):
        """
        Обучение модели
        """
        # self.model.load_weights(self.WEIGHTS_FILE)
        # return
        slicer_index = int(len(input_data) * LEARNING_PART)

        data_train_x = np.array(input_data[:slicer_index])
        data_test_x = np.array(input_data[slicer_index:])

        data_train_y = np.array(output_data[:slicer_index])
        data_test_y = np.array(output_data[slicer_index:])

        time_start = time.time()
        self.model.fit(
            x=data_train_x,
            y=data_train_y,
            epochs=EPOCHS,
            batch_size=16 * self.GPUS_COUNT,
            shuffle=True,
            callbacks=[self.loss_callback],
            verbose=False
        )
        # self.model.fit_generator(
        #     self.get_fit_data(data_train_x, data_train_y, batch_size=16),
        #     epochs=EPOCHS,
        #     shuffle=True,
        #     callbacks=[self.loss_callback]
        # )
        time_stop = time.time()

        test_accuracy = self.model.evaluate(x=data_test_x, y=data_test_y)
        print(f'Точность: {round(test_accuracy, 4)}')
        print(f'Время обучения {round((time_stop - time_start) / 60, 2)} м.')
        # self.model.save_weights(self.WEIGHTS_FILE)

    def show_loss_graphic(self):
        plt.plot(self.loss_callback.loss.keys(), self.loss_callback.loss.values())
        plt.show()

    def _predict_image(self, image: DatasetImage):
        _l = image.l
        _l = _l.reshape((1,) + _l.shape)
        predicted_ab = self.model.predict(_l)
        return predicted_ab.reshape(predicted_ab.shape[1:])

    def predict(self, image: DatasetImage):
        predicted_ab = self._predict_image(image)
        image.set_predicted_ab(predicted_ab)

    def predict_many(self, images):
        return [self.predict(image) for image in images]
