# -*- coding:utf-8 -*-
import os
import json
import numpy as np
from PIL import Image


def load_data(data_dir, flatten=False):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    meta_info = os.path.join(data_dir, 'meta.json')
    with open(meta_info, 'r') as f:
        meta = json.load(f)

    train_images, train_labels = _read_images_and_labels(train_dir, flatten=flatten, **meta)
    test_images, test_labels = _read_images_and_labels(test_dir, flatten=flatten, **meta)

    return (
        meta,
        DataSet(train_images, train_labels),
        DataSet(test_images, test_labels),
    )


def _read_images_and_labels(dir_name, flatten, ext='.png', **meta):
    images = []
    labels = []
    for fn in os.listdir(dir_name):
        if fn.endswith(ext):
            fd = os.path.join(dir_name, fn)
            images.append(_read_image(fd, flatten=flatten, **meta))
            labels.append(_read_label(fd, **meta))
    return np.array(images), np.array(labels)


def _read_image(filename, flatten, width, height, **extra_meta):
    im = Image.open(filename).convert('L').resize((width, height), Image.ANTIALIAS)

    data = np.asarray(im)
    if flatten:
        return data.reshape(width * height)

    return data


def _read_label(filename, label_choices, **extra_meta):
    basename = os.path.basename(filename)
    labels = basename.split('_')[0]

    data = []

    for c in labels:
        idx = label_choices.index(c)
        tmp = [0] * len(label_choices)
        tmp[idx] = 1
        data.extend(tmp)

    return data


class DataSet(object):
    """Provide `next_batch` method, which returns the next `batch_size` examples from this data set."""

    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):

        assert batch_size <= self._num_examples

        if self._index_in_epoch + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            self._index_in_epoch = 0

        if self._index_in_epoch == 0:
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

        # read next batch
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        return self._images[start:self._index_in_epoch], self._labels[start:self._index_in_epoch]


def display_debug_info(meta, train_data, test_data):
    print('%s Meta Info %s' % ('=' * 10, '=' * 10))
    for k, v in meta.items():
        print('%s: %s' % (k, v))
    print('=' * 30)

    print('train images: %s, labels: %s' % (train_data.images.shape, train_data.labels.shape))

    print('test images: %s, labels: %s' % (test_data.images.shape, test_data.labels.shape))


if __name__ == '__main__':
    import sys
    ret1 = load_data(data_dir=sys.argv[1])
    display_debug_info(*ret1)
