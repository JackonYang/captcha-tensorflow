# Captcha Solving Using TensorFlow


## Introduction

1. Solve captcha using TensorFlow.
2. Learn CNN and TensorFlow by a practical project.

Follow the steps,
run the code,
and it works!

the easiest model requires only 2 convolutional layers.
accuracy is 90%+

There are several more steps to put this prototype on production.

**Ping me for paid technical supports**.

[i@jackon.me](mailto:i@jackon.me)


## Tutorials for TensorFlow Beginners

this is an introduction to Tensorflow and TensorBoard.

skip this section if you are a experienced user.

#### Simple Softmax Model: 80%+ Accuracy

The dataset is similar to MNIST, The model is also similar to it.

1. Using only 1 softmax layer.
2. Accuracy: 80%+.
3. Training data: 20000 images.
4. Time cost: <3min. using GTX-1080.

```bash
# generating test data
$ python datasets/gen_captcha.py -d --npi 1 -n 2000
# run the model
$ python simple_softmax.py images/char-1-epoch-2000/
```

Output While Training:

```bash
$ time python simple_softmax.py images/char-1-epoch-2000/
data loaded
train images: 20000. test images: 4000
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

#### TensorBoard

Tensorboard is a suite of visualization tools.

Tensorboard can visualize TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it.

It is really cool and helpful.

some examples:

![](img-doc/m1-softmax-accuracy.png)
![](img-doc/m1-softmax-loss.png)
![](img-doc/m1-image-preview.png)
![](img-doc/m1-histograms.png)

**How doest it work?**

1. TensorFlow will write necessary info, defined by `tf.summary`, to log files.
2. TensorBoard will parse the logs and visualize the data.

Adding `tf.summary` to `simple_softmax.py`, we got `softmax_with_log.py`.

Run the model using the same training dataset

```bash
$ python softmax_with_log.py images/char-1-epoch-2000/
```

Launching TensorBoard in a new terminal

```bash
$ tensorboard --logdir=log
```

Navigate your web browser to `http://127.0.0.1:6006/` to view the TensorBoard.


## Solve Captcha Using CNN Model

#### 1-char Captcha

the same 2-layer CNN model as talked in MNIST tutorial

1. 2 Convolutional layers.
2. Accuracy: 97%
3. Training data: 20000 images.
4. Time cost: <4min. using GTX-1080.

```bash
# generating test data
$ python datasets/gen_captcha.py -d --npi 1 -n 2000
# run the model
$ python cnn_1_char.py --data_dir images/char-1-epoch-2000/
```

Output While Training:

```bash
$ $ time python cnn_1_char.py --data_dir images/char-1-epoch-2000/
data loaded
train images: 20000. test images: 4000
label_size: 10, image_size: 6000
...
step 9100, training accuracy = 100.00%, testing accuracy = 96.60%
step 9200, training accuracy = 100.00%, testing accuracy = 97.80%
step 9300, training accuracy = 100.00%, testing accuracy = 97.65%
step 9400, training accuracy = 100.00%, testing accuracy = 97.95%
step 9500, training accuracy = 100.00%, testing accuracy = 97.25%
step 9600, training accuracy = 98.00%, testing accuracy = 96.80%
step 9700, training accuracy = 100.00%, testing accuracy = 97.70%
step 9800, training accuracy = 100.00%, testing accuracy = 97.25%
step 9900, training accuracy = 100.00%, testing accuracy = 95.40%
testing accuracy = 97.85%

real	3m35.217s
user	3m4.168s
sys	0m31.036s
```

#### 2-char Captcha

Add 1 more layer, reshape the output before calculating Loss.

1. Accuracy: 97%
3. Training data: 45000 images.
4. Time cost: <5min. using GTX-1080.

```bash
# generating test data
$ python datasets/gen_captcha.py -d --npi=2 -n 500
# run the model
$ python cnn_n_char.py --data_dir=images/char-2-epoch-500/
```

Output While Training:

```bash
$ time python cnn_n_char.py --data_dir=images/char-2-epoch-500/
data loaded
train images: 45000. test images: 9000
label_size: 10, image_size: 8000
...
step 9100, training accuracy = 99.00%, testing accuracy = 96.45%
step 9200, training accuracy = 100.00%, testing accuracy = 97.28%
step 9300, training accuracy = 100.00%, testing accuracy = 96.75%
step 9400, training accuracy = 100.00%, testing accuracy = 96.68%
step 9500, training accuracy = 100.00%, testing accuracy = 96.55%
step 9600, training accuracy = 100.00%, testing accuracy = 96.40%
step 9700, training accuracy = 100.00%, testing accuracy = 95.58%
step 9800, training accuracy = 99.00%, testing accuracy = 96.70%
step 9900, training accuracy = 99.00%, testing accuracy = 96.68%
testing accuracy = 97.33%

real	4m49.509s
user	4m10.980s
sys	0m34.164s
```

#### 4-char Captcha

the same model as 2-char captcha.

only change the training dataset to 4-char ones.

```bash
# generating test data
$ python datasets/gen_captcha.py -d --npi=4 -n 6
# run the model
$ python cnn_n_char.py --data_dir images/char-4-epoch-6/
```

Output While Training:

```bash
$ python cnn_n_char.py --data_dir images/char-4-epoch-6/
data loaded
train images: 30240. test images: 5040
label_size: 10, image_size: 12000
...
step 9100, training accuracy = 100.00%, testing accuracy = 92.73%
step 9200, training accuracy = 98.00%, testing accuracy = 92.86%
step 9300, training accuracy = 97.50%, testing accuracy = 91.74%
step 9400, training accuracy = 99.00%, testing accuracy = 92.84%
step 9500, training accuracy = 99.00%, testing accuracy = 92.99%
step 9600, training accuracy = 99.00%, testing accuracy = 93.78%
step 9700, training accuracy = 100.00%, testing accuracy = 92.56%
step 9800, training accuracy = 99.50%, testing accuracy = 93.01%
step 9900, training accuracy = 99.00%, testing accuracy = 92.84%
testing accuracy = 93.24%
```

## Generate DataSet for Training

#### Usage

```bash
$ python datasets/gen_captcha.py  -h
usage: gen_captcha.py [-h] [-n N] [-t T] [-d] [-l] [-u] [--npi NPI]
                      [--data_dir DATA_DIR]

optional arguments:
  -h, --help           show this help message and exit
  -n N                 epoch number of character permutations.
  -t T                 ratio of test dataset.
  -d, --digit          use digits in dataset.
  -l, --lower          use lowercase in dataset.
  -u, --upper          use uppercase in dataset.
  --npi NPI            number of characters per image.
  --data_dir DATA_DIR  where data will be saved.
```

examples:

![](img-doc/data-set-example.png)

#### Example 1: 1 character per captcha, use digits only.

1 epoch will have 10 images, generate 2000 epoches for training.

generating the dataset:

```bash
$ python datasets/gen_captcha.py -d --npi 1 -n 2000
10 choices: 0123456789
generating 2000 epoches of captchas in ./images/char-1-epoch-2000/train
generating 400 epoches of captchas in ./images/char-1-epoch-2000/test
write meta info in ./images/char-1-epoch-2000/meta.json
```

preview the dataset:

```bash
$ python datasets/base.py images/char-1-epoch-2000/
========== Meta Info ==========
num_per_image: 1
label_choices: 0123456789
height: 100
width: 60
n_epoch: 2000
label_size: 10
==============================
train images: (20000, 100, 60), labels: (20000, 10)
test images: (4000, 100, 60), labels: (4000, 10)
```

#### Example 2: use digits/lower/upper cases, 2 digit per captcha image

1 epoch will have `62*61=3782` images, generate 10 epoches for training.

generating the dataset:

```bash
$ python datasets/gen_captcha.py -dlu --npi 2 -n 10
62 choices: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
generating 10 epoches of captchas in ./images/char-2-epoch-10/train
generating 2 epoches of captchas in ./images/char-2-epoch-10/test
write meta info in ./images/char-2-epoch-10/meta.json
```

preview the dataset:

```bash
========== Meta Info ==========
num_per_image: 2
label_choices: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
height: 100
width: 80
n_epoch: 10
label_size: 62
==============================
train images: (37820, 100, 80), labels: (37820, 124)
test images: (7564, 100, 80), labels: (7564, 124)
```


#### Example 2: use digits, 4 chars per captcha image

1 epoch has `10*9*8*7=5040` images, generate 10 epoches for training.

generating the dataset:

```bash
$ python datasets/gen_captcha.py -d --npi=4 -n 6
10 choices: 0123456789
generating 6 epoches of captchas in ./images/char-4-epoch-6/train
generating 1 epoches of captchas in ./images/char-4-epoch-6/test
write meta info in ./images/char-4-epoch-6/meta.json
```

preview the dataset:

```bash
$ python datasets/base.py images/char-4-epoch-6/
========== Meta Info ==========
num_per_image: 4
label_choices: 0123456789
height: 100
width: 120
n_epoch: 6
label_size: 10
==============================
train images: (30240, 100, 120), labels: (30240, 40)
test images: (5040, 100, 120), labels: (5040, 40)
```
