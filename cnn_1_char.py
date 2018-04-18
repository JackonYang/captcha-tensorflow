# -*- coding:utf-8 -*-
import argparse
import datetime
import sys
import tensorflow as tf

import datasets.base as input_data

MAX_STEPS = 10000
BATCH_SIZE = 50

LOG_DIR = 'log/cnn1-run-%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

FLAGS = None


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    # load data
    meta, train_data, test_data = input_data.load_data(FLAGS.data_dir, flatten=False)
    print('data loaded')
    print('train images: %s. test images: %s' % (train_data.images.shape[0], test_data.images.shape[0]))

    LABEL_SIZE = meta['label_size']
    IMAGE_HEIGHT = meta['height']
    IMAGE_WIDTH = meta['width']
    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))

    # variable in the graph for input data
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH])
        y_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])

        # must be 4-D with shape `[batch_size, height, width, channels]`
        x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x_image, max_outputs=LABEL_SIZE)

    # define the model
    with tf.name_scope('convolution-layer-1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('convolution-layer-2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('densely-connected'):
        W_fc1 = weight_variable([IMAGE_WIDTH * IMAGE_HEIGHT * 4, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_WIDTH*IMAGE_HEIGHT*4])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        # To reduce overfitting, we will apply dropout before the readout layer
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('readout'):
        W_fc2 = weight_variable([1024, LABEL_SIZE])
        b_fc2 = bias_variable([LABEL_SIZE])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Define loss and optimizer
    # Returns:
    # A 1-D `Tensor` of length `batch_size`
    # of the same type as `logits` with the softmax cross entropy loss.
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            # -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        variable_summaries(cross_entropy)

    # forword prop
    with tf.name_scope('forword-prop'):
        predict = tf.argmax(y_conv, axis=1)
        expect = tf.argmax(y_, axis=1)

    # evaluate accuracy
    with tf.name_scope('evaluate_accuracy'):
        correct_prediction = tf.equal(predict, expect)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_summaries(accuracy)

    with tf.Session() as sess:

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)

        tf.global_variables_initializer().run()

        # Train
        for i in range(MAX_STEPS):
            batch_xs, batch_ys = train_data.next_batch(BATCH_SIZE)

            step_summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            train_writer.add_summary(step_summary, i)

            if i % 100 == 0:
                # Test trained model
                valid_summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                train_writer.add_summary(valid_summary, i)

                # final check after looping
                test_x, test_y = test_data.next_batch(2000)
                test_summary, test_accuracy = sess.run([merged, accuracy], feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
                test_writer.add_summary(test_summary, i)

                print('step %s, training accuracy = %.2f%%, testing accuracy = %.2f%%' % (i, train_accuracy * 100, test_accuracy * 100))

        train_writer.close()
        test_writer.close()

        # final check after looping
        test_x, test_y = test_data.next_batch(2000)
        test_accuracy = accuracy.eval(feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
        print('testing accuracy = %.2f%%' % (test_accuracy * 100, ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='images/char-1-epoch-2000/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
