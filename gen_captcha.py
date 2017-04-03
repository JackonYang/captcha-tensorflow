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


def _gen_captcha(n=1000, num_per_image=1, img_dir='images', d=True, l=False, u=False):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=40 + 20 * num_per_image, height=100)
    print 'generating %s captchas in %s' % (n, img_dir)

    if not (d or l or u):
        print 'Digit, lower case and upper case cannot be all False.'
        return
    else:
        label_list = []
        if d:
            label_list = label_list + list(range(10))
        if l:
            label_list = label_list + list(string.ascii_lowercase)
        if u:
            label_list = label_list + list(string.ascii_uppercase)
        label_set = map(str, label_list)
        for _ in range(n):
            for i in itertools.permutations(label_set, num_per_image):
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
        '-d',
        default=True,
        type=bool,
        help='use digits in labels.')
    parser.add_argument(
        '-l',
        default=False,
        type=bool,
        help='use lowercase characters in labels.')
    parser.add_argument(
        '-u',
        default=False,
        type=bool,
        help='use uppercase characters in labels.')
    parser.add_argument(
        '--npi',
        default=1,
        type=int,
        help='number of characters per image.')

    FLAGS, unparsed = parser.parse_known_args()

    train_number = FLAGS.n
    test_number = int(FLAGS.n * FLAGS.t)
    digit = FLAGS.d
    lower = FLAGS.l
    upper = FLAGS.u
    num_per_image=FLAGS.npi

    _gen_captcha(n=train_number, num_per_image=num_per_image,
            img_dir='images/char'+str(num_per_image)+'/train', d=digit, l=lower, u=upper)
    _gen_captcha(n=test_number, num_per_image=num_per_image,
            img_dir='images/char'+str(num_per_image)+'/test', d=digit, l=lower, u=upper)
