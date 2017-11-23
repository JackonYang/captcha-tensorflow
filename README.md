# captcha-tensorflow

the 1st tensorflow project.

Solve captcha using TensorFlow


## Quick Start

#### Simple-softmax: 1 个字符

1. 1 个 softmax 层
2. 正确率 90%
3. 10000 张图片，数字 0-9
4. 训练时间 3min. GTX 1080


生成测试数据, 1000 组, 纯数字

```bash
$ python gen_captcha.py -n 1000 -d
```

训练

```bash
$ time python simple_softmax.py
data loaded
train images: 10000. test images: 2000
label_size: 10, image_size: 6000
...
step = 9100, accuracy = 91.10%
step = 9200, accuracy = 91.40%
step = 9300, accuracy = 92.00%
step = 9400, accuracy = 91.40%
step = 9500, accuracy = 91.35%
step = 9600, accuracy = 90.80%
step = 9700, accuracy = 91.60%
step = 9800, accuracy = 91.65%
step = 9900, accuracy = 90.50%
testing accuracy = 91.05%

real2m46.478s
user2m29.704s
sys0m17.828s
```

#### tensorboard


基本的原理：

tensorflow 执行时，写 log 文件，
tensorboard 解析 log 并做数据可视化。

定义 graph 的时候，
用 tf.summary 定义需要写入日志的变量值和格式。

代码：`softmax_with_log.py`


```bash
$ python softmax_with_log.py
```

在另外 1 个 terminal 中执行

```bash
$ tensorboard --logdir=log
```

浏览器中打开 `http://127.0.0.1:6006/`

![](img-doc/m1-softmax-accuracy.png)
![](img-doc/m1-softmax-loss.png)
![](img-doc/m1-image-preview.png)
![](img-doc/m1-histograms.png)


#### 2 层 Convolutional: 1 个字符

1. 2 个 Convolutional 层
2. 正确率 99%
3. 10000 张图片，数字 0-9
4. 训练时间 5min. 10k steps，GTX 1080

```bash
$ time python conv_captcha.py
data loaded
train images: 10000. test images: 2000
label_size: 10, image_size: 6000
...
step 9100, training accuracy = 100.00%, testing accuracy = 98.90%
step 9200, training accuracy = 100.00%, testing accuracy = 98.80%
step 9300, training accuracy = 100.00%, testing accuracy = 98.90%
step 9400, training accuracy = 100.00%, testing accuracy = 98.80%
step 9500, training accuracy = 100.00%, testing accuracy = 98.90%
step 9600, training accuracy = 100.00%, testing accuracy = 98.60%
step 9700, training accuracy = 100.00%, testing accuracy = 98.65%
step 9800, training accuracy = 100.00%, testing accuracy = 98.95%
step 9900, training accuracy = 100.00%, testing accuracy = 98.95%
testing accuracy = 99.15%

real	4m44.143s
user	3m40.896s
sys	0m33.492s
```


## 训练数据

每一个 dataset 分 train 和 test 2 个目录存放图片数据。
根目录下的 meta.json 存放参数信息。

meta.json 的例子

```javascript
{
    "num_per_image": 1,
    "n_train": 1,
    "label_choices": "0123456789",
    "n_test": 1,
    "width": 60,
    "height": 100,
    "label_size": 10
}
```

图片的例子

![](img-doc/data-set-example.png)


#### 生成

使用 python 的 captcha package 生成测试数据

查看用法说明

```bash
$ python gen_captcha.py -h
usage: gen_captcha.py [-h] [-n N] [-t T] [-d] [-l] [-u] [--npi NPI]

optional arguments:
  -h, --help   show this help message and exit
  -n N         number of captchas permutations
  -t T         ratio of test / train.
  -d, --digit  use digits in labels.
  -l, --lower  use lowercase characters in labels.
  -u, --upper  use uppercase characters in labels.
  --npi NPI    number of characters per image.
```

例如，生成包含数字 + 小写字母的验证码，每张图片包含 2 个字符，
10 组训练数据，另外生成 10% 的测试数据

```bash
$ python gen_captcha.py -dl --npi 2 -n 10 -t 0.1
36 choices: 0123456789abcdefghijklmnopqrstuvwxyz
generating 10 groups of captchas in images/char-2-groups-10/train
generating 1 groups of captchas in images/char-2-groups-10/test
write meta info in images/char-2-groups-10/meta.json
```

用时约 1 min。生成的图片数量如下

```bash
$ ls images/char-2-groups-10/train | wc -l
   12600
$ ls images/char-2-groups-10/test/ | wc -l
    1260
```
