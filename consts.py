import os

# Размер картинки в пикселях (w,h)
IMAGE_SIZE = 64

# Количество изображений в выборке (макс 100 000)
IMAGES_COUNT = 3500

# Обучающая часть в долях
LEARNING_PART = 0.7

# Количество эпох
EPOCHS = 100

# Размер batch
BATCH_SIZE = 128

# Количество видеокарт
GPUS_COUNT = 1

# Использовать тензорные ядра
USE_TPU = False

# --------------------------------------------------------------------------------------------------------------------

# Директория с изображениями
IMAGES_PATH = 'images/'

# Оригинальные картинки
IMAGES_ORIGINAL_PATH = os.path.join(IMAGES_PATH, 'original/')

# Картинки с изменённым размером
IMAGES_RESIZED_PATH = os.path.join(IMAGES_PATH, 'resized/', f'{IMAGE_SIZE}x{IMAGE_SIZE}/')
