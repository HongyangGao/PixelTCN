import scipy
import scipy.misc
import numpy as np


def center_crop(image, pre_height, pre_width, height, width):
    h, w = image.shape[:2]
    j, i = int((h - pre_height)/2.), int((w - pre_width)/2.)
    return scipy.misc.imresize(
        image[j:j+pre_height, i:i+pre_width], [height, width])


def transform(image, pre_height, pre_width, height, width, is_crop):
    if is_crop:
        new_image = center_crop(image, pre_height, pre_width, height, width)
    else:
        new_image = scipy.misc.imresize(image, [height, width])
    return np.array(new_image)/127.5 - 1.


def imread(path, is_grayscale=False):
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    return scipy.misc.imread(path).astype(np.float)


def imsave(image, path):
    scipy.misc.imsave(path, image)


def get_images(paths, pre_height, pre_width, height, width,
               is_crop=False, is_grayscale=False):
    images = []
    for path in paths:
        image = imread(path, is_grayscale)
        new_image = transform(
            image, pre_height, pre_width, height, width, is_crop)
        images.append(new_image)
    return np.array(images).astype(np.float32)
