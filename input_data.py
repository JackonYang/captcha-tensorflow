# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np
import json


def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    meta_info = os.path.join(data_dir, 'meta.json')
    with open(meta_info, 'r') as f:
        meta = json.load(f)

    return (
        meta,
        DataSet(
            *_read_images_and_labels(train_dir, **meta)),
        DataSet(
            *_read_images_and_labels(test_dir, **meta)),
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


def _read_images_and_labels(dir_name, ext='.png', **meta):
    images = []
    labels = []
    for fn in os.listdir(dir_name):
        if fn.endswith(ext):
            fd = os.path.join(dir_name, fn)
            images.append(_read_image(fd, **meta))
            labels.append(_read_lable(fd, **meta))
    return np.array(images), np.array(labels)


def _read_image(filename, **extra_meta):
    im = Image.open(filename).convert('L')
    data = np.asarray(im)
    return data


def _read_lable(filename, label_choices, **extra_meta):
    basename = os.path.basename(filename)
    idx = label_choices.index(basename.split('_')[0])
    data = np.zeros(len(label_choices))
    data[idx] = 1
    return data


if __name__ == '__main__':
    meta, train_data, test_data = load_data('images/char-1-groups-1000/')

    print meta

    print 'train images: %s, labels: %s' % (train_data.images.shape, train_data.labels.shape)

    print 'test images: %s, labels: %s' % (test_data.images.shape, test_data.labels.shape)

    batch_xs, batch_ys = train_data.next_batch(100)
    print 'batch images: %s, labels: %s' % (batch_xs.shape, batch_ys.shape)
