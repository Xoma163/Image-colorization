import os

# Размер картинки в пикселях (w,h)
IMAGE_SIZE = 64

# Количество изображений в выборке (макс 100 000)
IMAGES_COUNT = 25000

# Обучающая часть в долях
LEARNING_PART = 0.7

# Количество эпох
EPOCHS = 100

# Размер batch
BATCH_SIZE = 128

# Количество видеокарт
GPUS_COUNT = 1

# --------------------------------------------------------------------------------------------------------------------

# Директория с изображениями
IMAGES_PATH = 'images/'

# Оригинальные картинки
IMAGES_ORIGINAL_PATH = os.path.join(IMAGES_PATH, 'original/')

# Картинки с изменённым размером
IMAGES_RESIZED_PATH = os.path.join(IMAGES_PATH, 'resized/', f'{IMAGE_SIZE}x{IMAGE_SIZE}/')

# Директория с логами
LOGS_DIR = 'logs/'

LOGS_INFO_FILE = os.path.join(LOGS_DIR, 'info.log')

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(IMAGES_RESIZED_PATH, exist_ok=True)
