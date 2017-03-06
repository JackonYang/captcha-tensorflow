import argparse
import sys

import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    image_px = 60 * 100  # image size: width * height
    y_len = 10  # 0 - 9

    # load data
    train_data, test_data = input_data.load_one_char(FLAGS.data_dir)
    print 'data loaded'

    # Create the model
    x = tf.placeholder(tf.float32, [None, image_px])
    W = tf.Variable(tf.zeros([image_px, y_len]))
    b = tf.Variable(tf.zeros([y_len]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, y_len])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Train
        for _ in range(10000):
            batch_xs, batch_ys = train_data.next_batch(1000)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if _ % 100 == 0:
                # Test trained model
                correct_prediction = tf.equal(tf.argmax(y, 1),
                                              tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))
                r = sess.run(accuracy, feed_dict={x: test_data.images,
                                                  y_: test_data.labels})
                print 'step %s: %.2f%%' % (_, r * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='images/one-char',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
