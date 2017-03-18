# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np

IMAGE_SIZE_1CHAR = 60 * 100  # width * height


def load_data_1char(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    return (
        DataSet(*_load_data(train_dir, IMAGE_SIZE_1CHAR)),
        DataSet(*_load_data(test_dir, IMAGE_SIZE_1CHAR)),
    )


class DataSet:
    """提供 next_batch 方法"""
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels

        self._num_examples = images.shape[0]

        self.ptr = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, size=100, shuffle=True):
        if self.ptr + size > self._num_examples:
            self.ptr = 0

        if self.ptr == 0:
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]

        self.ptr += size
        return (
            self._images[self.ptr - size: self.ptr],
            self._labels[self.ptr - size: self.ptr],
        )


def _load_data(dir_name, size=None, ext='.png'):
    images = []
    labels = []
    for fn in os.listdir(dir_name):
        if fn.endswith(ext):
            fd = os.path.join(dir_name, fn)
            images.append(load_image(fd, size))
            labels.append(load_label(fd))
    return np.vstack(images), np.vstack(labels)


def load_image(filename, size=None):
    im = Image.open(filename).convert('L')
    data = np.asarray(im)

    if size:
        return data.reshape(size)
    return data


def load_label(filename, size=10):
    basename = os.path.basename(filename)
    num = int(basename.split('_')[0])
    data = np.zeros(size)
    data[num] = 1
    return data


if __name__ == '__main__':
    train_data, test_data = load_data_1char('images/one-char')

    print 'train images: %s, labels: %s' % (train_data.images.shape, train_data.labels.shape)

    print 'test images: %s, labels: %s' % (test_data.images.shape, test_data.labels.shape)

    batch_xs, batch_ys = train_data.next_batch(100)
    print 'batch images: %s, labels: %s' % (batch_xs.shape, batch_ys.shape)
