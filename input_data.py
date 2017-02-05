import os
from PIL import Image
import numpy as np


class DataSet:
    def __init__(self, xs, ys):
        self._xs = xs
        self._ys = ys

        self.ptr = 0

    @property
    def images(self):
        return self._xs

    @property
    def labels(self):
        return self._ys

    def next_batch(self, size=100):
        self.ptr += size
        return (
            self._xs[self.ptr - size: self.ptr],
            self._ys[self.ptr - size: self.ptr],
        )


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


def load_one_char(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    return (
        DataSet(*load_data_set(train_dir)),
        DataSet(*load_data_set(test_dir)),
    )


if __name__ == '__main__':
    train_data, test_data = load_one_char('images/one-char')

    print 'train data'
    print train_data.images.shape, train_data.labels.shape

    print 'test data'
    print test_data.images.shape, test_data.labels.shape

    print 'batch data'
    batch_xs, batch_ys = train_data.next_batch(20)
    print batch_xs.shape, batch_ys.shape
