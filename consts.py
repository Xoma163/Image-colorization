import os

# Размер картинки в пикселях (w,h)
IMAGE_SIZE = 128
# Количество изображений в выборке (макс 100 000)
IMAGES_COUNT = 1000

# Обучающая часть в долях
LEARNING_PART = 0.7

# Количество эпох
EPOCHS = 200

# --------------------------------------------------------------------------------------------------------------------

# Директория с изображениями
IMAGES_PATH = 'images/'

# Оригинальные картинки
IMAGES_ORIGINAL_PATH = os.path.join(IMAGES_PATH, 'original/')

# Картинки с изменённым размером
IMAGES_RESIZED_PATH = os.path.join(IMAGES_PATH, 'resized/', f'{IMAGE_SIZE}x{IMAGE_SIZE}/')
