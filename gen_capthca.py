# -*- coding:utf-8 -*-
import os
import uuid
from captcha.image import ImageCaptcha


def one_char(loop=1000, img_dir='images'):
    image = ImageCaptcha(width=60, height=100)
    for _ in range(loop):
        if _ % 100 == 0:
            print '%s looping...' % _
        for i in range(10):
            fn = os.path.join(img_dir, '%s_%s.png' % (i, uuid.uuid4()))
            image.write(str(i), fn)


if __name__ == '__main__':
    one_char()
