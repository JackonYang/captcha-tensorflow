# -*- coding:utf-8 -*-
import argparse
import string
import os
import uuid
from captcha.image import ImageCaptcha

import itertools


CHOICES = map(str, list(range(10)) + list(string.ascii_lowercase))


def one_char(n=1000, img_dir='images'):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=60, height=100)
    print 'generating %s captchas in %s' % (n, img_dir)

    for _ in range(n):
        for i in CHOICES:
            fn = os.path.join(img_dir, '%s_%s.png' % (i, uuid.uuid4()))
            image.write(str(i), fn)


def _gen_captcha(n, num_per_image, img_dir='images'):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=40 + 20 * num_per_image, height=100)
    print 'generating %s captchas in %s' % (n, img_dir)

    for _ in range(n):
        for i in itertools.permutations(CHOICES, num_per_image):
            captcha = ''.join(i)
            fn = os.path.join(img_dir, '%s_%s.png' % (captcha, uuid.uuid4()))
            image.write(captcha, fn)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        default=1000,
        type=int,
        help='number of captchas for one integer.')
    parser.add_argument(
        '-t',
        default=0.2,
        type=float,
        help='ratio of test / train.')

    FLAGS, unparsed = parser.parse_known_args()
    train_number = FLAGS.n
    test_number = int(FLAGS.n * FLAGS.t)

    _gen_captcha(train_number, 4, 'images/char4/train')
    _gen_captcha(test_number, 4, 'images/char4/test')
