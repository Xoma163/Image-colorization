import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, layers, models
from tensorflow.python.data.experimental import AutoShardPolicy
from tensorflow.python.keras.callbacks import Callback

from ImageHandler import DatasetImage
from consts import LEARNING_PART, EPOCHS, IMAGE_SIZE, GPUS_COUNT, BATCH_SIZE
from utils import CyclePercentWriter, lead_time_writer, get_time_str


class LossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.loss = {}
        self.cpw = CyclePercentWriter(EPOCHS, per=10)
        self.last_epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.loss[epoch] = (logs['loss'])
        if self.cpw.check(epoch):
            time_now = time.time()
            # Предсказание времени работы
            epochs_remain = EPOCHS - epoch
            epochs_per_trigger = self.cpw.per / 100 * EPOCHS
            train_time = time_now - self.last_epoch_start
            remaining_time = epochs_remain / epochs_per_trigger * train_time
            print(f"Эпоха {epoch}/{EPOCHS}. "
                  f"Ошибка {round(logs['loss'], 5)}. "
                  f"Время обучения {get_time_str(train_time)}. "
                  f"Осталось ~{get_time_str(remaining_time)}")
            self.last_epoch_start = time_now


class NeuralNetwork:
    """
    Свёрточная нейронная сеть
    """
    WEIGHTS_FILE = 'model/weights'

    # https://github.com/keras-team/keras/issues/8123

    @staticmethod
    def get_compiled_model():
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
            layers.Conv2D(1, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D(2),
        ])
        model.compile(optimizer=optimizers.Adam(), loss='mse')
        return model

    def __init__(self):
        print(f"Found devices:\n"
              f"{[x.name for x in tf.config.list_logical_devices()]}")

        if GPUS_COUNT == 1:
            self.model = self.get_compiled_model()
        else:
            strategy = MirroredStrategy()
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

            with strategy.scope():
                self.model = self.get_compiled_model()

        # self.model.summary()
        self.loss_callback = LossCallback()

    @staticmethod
    def prepare_datasets(input_data, output_data):
        slicer_index = int(len(input_data) * LEARNING_PART)

        train_data_x = np.array(input_data[:slicer_index])
        test_data_x = np.array(input_data[slicer_index:])
        del input_data

        train_data_y = np.array(output_data[:slicer_index])
        test_data_y = np.array(output_data[slicer_index:])
        del output_data

        train_data = tf.data.Dataset.from_tensor_slices((train_data_x, train_data_y))
        test_data = tf.data.Dataset.from_tensor_slices((test_data_x, test_data_y))

        train_data = train_data.shuffle(len(train_data_x), reshuffle_each_iteration=True)
        train_data = train_data.batch(BATCH_SIZE)  # , drop_remainder=True)

        test_data = test_data.batch(BATCH_SIZE)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA

        train_data = train_data.with_options(options)
        test_data = test_data.with_options(options)
        return train_data, test_data

    @lead_time_writer
    def train(self, input_data, output_data):
        """
        Обучение модели
        """
        # self.model.load_weights(self.WEIGHTS_FILE)
        # return
        train_data, test_data = self.prepare_datasets(input_data, output_data)
        start_time = time.time()
        self.model.fit(
            train_data,
            epochs=EPOCHS,
            shuffle=True,
            callbacks=[self.loss_callback],
            verbose=False,
            # validation_data=data_test
        )
        end_time = time.time()
        print(f"АААААААААА {get_time_str(end_time - start_time)}")

        # test_accuracy = self.model.evaluate(x=data_test_x, y=data_test_y, verbose=False)
        # print(f'Точность: {round(test_accuracy, 5)}')
        self.model.save_weights(self.WEIGHTS_FILE)

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
