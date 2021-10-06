# -*- coding:utf-8 -*-
import argparse
import json
import random
import string
import os
import shutil
import uuid
from captcha.image import ImageCaptcha

import itertools

FLAGS = None
META_FILENAME = 'meta.json'


def get_choices():
    choices = [
        (FLAGS.digit, map(str, range(10))),
        (FLAGS.lower, string.ascii_lowercase),
        (FLAGS.upper, string.ascii_uppercase),
        ]
    return tuple([i for is_selected, subset in choices for i in subset if is_selected])


def _gen_captcha(img_dir, num_per_image, n, width, height, choices, max_images_count):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=width, height=height)

    remain_count = max_images_count
    epoche_count = len(list(itertools.permutations(choices, num_per_image)))

    print('generating %s epoches of captchas in %s.' % (n, img_dir))

    for _ in range(n):
        samples = itertools.permutations(choices, num_per_image)
        if remain_count > 0 and remain_count < epoche_count:
            print('only %s records used in epoche %s. epoche_count: %s' % (remain_count, _+1, epoche_count))
            samples = random.sample(list(samples), remain_count)

        for i in samples:
            captcha = ''.join(i)
            fn = os.path.join(img_dir, '%s_%s.png' % (captcha, uuid.uuid4()))
            image.write(captcha, fn)
        if n < 20:
            print('(%s/%s) epoches finished' % (_+1, n))


def build_file_path(x):
    return os.path.join(FLAGS.data_dir, 'char-%s-epoch-%s' % (FLAGS.npi, FLAGS.n), x)


def gen_dataset():
    n_epoch = FLAGS.n
    num_per_image = FLAGS.npi
    test_ratio = FLAGS.t
    max_images_count = FLAGS.c

    choices = get_choices()

    width = 40 + 20 * num_per_image
    height = 100

    # meta info
    meta = {
        'num_per_image': num_per_image,
        'label_size': len(choices),
        'label_choices': ''.join(choices),
        'n_epoch': n_epoch,
        'width': width,
        'height': height,
    }

    print('%s choices: %s' % (len(choices), ''.join(choices) or None))

    _gen_captcha(build_file_path('train'), num_per_image, n_epoch, width, height, choices=choices, max_images_count=max_images_count)

    if test_ratio > 0:
        _gen_captcha(build_file_path('test'), num_per_image, max(1, int(n_epoch * test_ratio)), width, height, choices=choices, max_images_count=max(1, int(max_images_count * test_ratio)))

    meta_filename = build_file_path(META_FILENAME)
    with open(meta_filename, 'w') as f:
        json.dump(meta, f, indent=4)
    print('write meta info in %s' % meta_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        default=1,
        type=int,
        help='epoch number of character permutations.')

    parser.add_argument(
        '-c',
        default=0,
        type=int,
        help='max count of images to generate. default unlimited')

    parser.add_argument(
        '-t',
        default=0,
        type=float,
        help='ratio of test dataset.')

    parser.add_argument(
        '-d', '--digit',
        action='store_true',
        help='use digits in dataset.')
    parser.add_argument(
        '-l', '--lower',
        action='store_true',
        help='use lowercase in dataset.')
    parser.add_argument(
        '-u', '--upper',
        action='store_true',
        help='use uppercase in dataset.')
    parser.add_argument(
        '--npi',
        default=1,
        type=int,
        help='number of characters per image.')
    parser.add_argument(
        '--data_dir',
        default='./images',
        type=str,
        help='where data will be saved.')

    FLAGS, unparsed = parser.parse_known_args()

    gen_dataset()
