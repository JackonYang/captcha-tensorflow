import os
from PIL import Image
import numpy as np


def load_image(filename, size=None):
    im = Image.open(filename).convert('L')
    data = np.asarray(im)
    if size:
        return data.reshape(1 * size)
    else:
        return data


def load_label(filename, size=10):
    basename = os.path.basename(filename)
    num = int(basename.split('_')[0])
    data = np.zeros(1 * size)
    data[num] = 1
    return data


def load_data_set(dir_name, size=60*100, ext='.png'):
    images = []
    labels = []
    for fn in os.listdir(dir_name):
        if fn.endswith(ext):
            fd = os.path.join(dir_name, fn)
            images.append(load_image(fd, size))
            labels.append(load_label(fd))
    return np.vstack(images), np.vstack(labels)


if __name__ == '__main__':
    images, labels = load_data_set('images/one-char/train')
    print images.shape, labels.shape
