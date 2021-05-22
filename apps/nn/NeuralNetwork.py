import os

from .ImageHandler import DatasetImage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, layers, models
from tensorflow.python.data.experimental import AutoShardPolicy
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K

from .consts import LEARNING_PART, EPOCHS, IMAGE_SIZE, GPUS_COUNT, BATCH_SIZE
from .utils import CyclePercentWriter, lead_time_writer, get_time_str, get_logger

logger = get_logger(__name__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class LossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.loss_train = {}
        self.loss_test = {}
        self.cpw = CyclePercentWriter(EPOCHS, per=5)
        self.last_epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.loss_train[epoch + 1] = (logs['loss'])
        self.loss_test[epoch + 1] = (logs['val_loss'])
        if self.cpw.check(epoch + 1):
            time_now = time.time()
            # Предсказание времени работы
            epochs_remain = EPOCHS - epoch
            epochs_per_trigger = self.cpw.per / 100 * EPOCHS
            train_time = time_now - self.last_epoch_start
            remaining_time = epochs_remain / epochs_per_trigger * train_time
            logger.debug(
                f"Эпоха {epoch + 1}/{EPOCHS}. "
                f"Ошибка обучения {round(logs['loss'], 5)}. "
                f"Ошибка тестирования {round(logs['val_loss'], 5)}. "
                f"Время обучения {get_time_str(train_time)}. "
                f"Осталось ~{get_time_str(remaining_time)}"
            )
            self.last_epoch_start = time_now


class NeuralNetwork:
    """
    Свёрточная нейронная сеть
    """
    # WEIGHTS_FILE = 'apps/nn/model/weights'
    WEIGHTS_FILE = 'apps/nn/very_good_model/weights'

    # https://github.com/keras-team/keras/issues/8123

    # def build_lab(self, ab):
    #     l = np.ones((64, 64, 1)) / 2
    #     lab = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
    #     ab = ab.numpy()[0]
    #     lab[:, :, 0] = l[:, :, 0]
    #     lab[:, :, 1:] = ab
    #     return lab
    #
    # def ssim_loss(self, ab_true, ab_pred):
    #     # return tf.reduce_mean(tf.image.ssim(ab_true, ab_pred, 1.0))
    #     # rgbs_true = []
    #     # rgbs_pred = []
    #     # for i in range(len(ab_true)):
    #     #     lab_true = self.build_lab(ab_true[i])
    #     #     lab_pred = self.build_lab(ab_pred[i])
    #     #     rgb_true = lab2rgb(lab_true)
    #     #     rgb_pred = lab2rgb(lab_pred)
    #     #     rgbs_true.append(rgb_true)
    #     #     rgbs_pred.append(rgb_pred)
    #     #
    #     # rgbs_true = np.array(rgbs_true)
    #     # rgbs_pred = np.array(rgbs_pred)
    #
    #     # ssim = tf.image.ssim(rgbs_true, rgbs_pred, 1.0)
    #     ssim2 = tf.image.ssim(ab_true, ab_pred, 1.0)
    #     # reduce_mean = tf.reduce_mean(ssim)
    #     reduce_mean2 = tf.reduce_mean(ssim2)
    #     # print(type(ssim))
    #     # print(ssim.shape)
    #     # print(reduce_mean)
    #     # print('-----')
    #     # print(type(ssim2))
    #     # print(ssim2.shape)
    #     # print(reduce_mean2)
    #
    #     return reduce_mean2

    def get_compiled_model(self):
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
            layers.UpSampling2D(2),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D(2),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D(2),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(2, (3, 3), activation='relu', padding='same'),

        ])
        model.compile(
            optimizer=optimizers.Adam(),
            loss='mse',
            # run_eagerly=True,
            # metrics=['accuracy']
        )
        return model

    def __init__(self):
        logger.info(f"Found devices: {[x.name for x in tf.config.list_logical_devices()]}")



        if GPUS_COUNT == 1:
            self.model = self.get_compiled_model()
        else:
            strategy = MirroredStrategy()
            logger.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

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

    def load_model(self):
        self.model.load_weights(self.WEIGHTS_FILE)
        return

    @lead_time_writer
    def train(self, input_data, output_data):
        """
        Обучение модели
        """
        train_data, test_data = self.prepare_datasets(input_data, output_data)
        K.set_value(self.model.optimizer.learning_rate, 0.00075)  # Ошибка обучения 0.0005. Ошибка тестирования 0.00299.
        self.model.fit(
            train_data,
            epochs=EPOCHS,
            shuffle=True,
            callbacks=[self.loss_callback],
            verbose=False,
            validation_data=test_data
        )
        self.model.save_weights(self.WEIGHTS_FILE)
        self.show_loss_graphics()

    def show_loss_graphics(self):
        offset = 0
        max_y = 0.005

        axes = plt.gca()
        axes.set_ylim([0, max_y])

        plt.title('График величины ошибки от эпохи')
        plt.xlabel('Эпоха')
        plt.ylabel('Ошибка')
        plt.plot(
            list(self.loss_callback.loss_train.keys())[offset:],
            list(self.loss_callback.loss_train.values())[offset:],
            label='Обучение'
        )
        plt.plot(
            list(self.loss_callback.loss_test.keys())[offset:],
            list(self.loss_callback.loss_test.values())[offset:],
            label='Тест'
        )
        plt.legend()
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
