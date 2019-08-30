import os

IMAGES_BEFORE = 'images_before'
IMAGES_AFTER = 'images_after/original'
IMAGES_AFTER_BLACK_WHITE = 'images_after/black-white'

from PIL import Image


def convert_image(input_image_path, output_image_path, output_image_bw_path, size):
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(size)
    resized_image.save(output_image_path)
    print('done resize')

    bw_image = resized_image.convert('L')  # конвертация изображения в серые цвета
    bw_image.save(output_image_bw_path)
    print('done convert to black-white')


if __name__ == '__main__':
    files = os.listdir(IMAGES_BEFORE)
    for file in files:
        convert_image(input_image_path='%s/%s' % (IMAGES_BEFORE, file),
                      output_image_path='%s/%s' % (IMAGES_AFTER, file),
                      output_image_bw_path='%s/%s' % (IMAGES_AFTER_BLACK_WHITE, file),
                      size=(100, 100))
