# -*- coding:utf-8 -*-
import argparse
import string
import os
import uuid
from captcha.image import ImageCaptcha

import itertools

FLAGS = None

CHOICES = map(str, list(range(10)) + list(string.ascii_lowercase))


def get_choices():
    cate_map = [
        (FLAGS.digit, map(str, range(10))),
        (FLAGS.lower, string.ascii_lowercase),
        (FLAGS.upper, string.ascii_uppercase),
        ]
    return tuple([i for _flag, choices in cate_map for i in choices if _flag])


def one_char(n=1000, img_dir='images'):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=60, height=100)
    print 'generating %s captchas in %s' % (n, img_dir)

    for _ in range(n):
        for i in CHOICES:
            fn = os.path.join(img_dir, '%s_%s.png' % (i, uuid.uuid4()))
            image.write(str(i), fn)


def _gen_captcha(n=1000, num_per_image=1, img_dir='images', choices=[]):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=40 + 20 * num_per_image, height=100)
    print 'generating %s captchas in %s' % (n, img_dir)

    for _ in range(n):
        for i in itertools.permutations(choices, num_per_image):
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
    parser.add_argument(
        '-d', '--digit',
        action='store_true',
        help='use digits in labels.')
    parser.add_argument(
        '-l', '--lower',
        action='store_true',
        help='use lowercase characters in labels.')
    parser.add_argument(
        '-u', '--upper',
        action='store_true',
        help='use uppercase characters in labels.')
    parser.add_argument(
        '--npi',
        default=1,
        type=int,
        help='number of characters per image.')

    FLAGS, unparsed = parser.parse_known_args()

    train_number = FLAGS.n
    test_number = int(FLAGS.n * FLAGS.t)
    num_per_image = FLAGS.npi

    a = get_choices()
    print len(a)
    print a
    # _gen_captcha(n=train_number, num_per_image=num_per_image, img_dir='images/char'+str(num_per_image)+'/train', choices=get_choices())
    # _gen_captcha(n=test_number, num_per_image=num_per_image, img_dir='images/char'+str(num_per_image)+'/test', choices=get_choices())
