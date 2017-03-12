import argparse
import sys
import numpy as np

import input_data

import tensorflow as tf

FLAGS = None

log_dir = 'tf-log'


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        tf.summary.historgram('historgram', var)


def main(_):
    image_px = 60 * 100  # image size: width * height
    y_len = 10  # 0 - 9

    # load data
    train_data, test_data = input_data.load_one_char(FLAGS.data_dir)
    print 'data loaded'

    with tf.name_scope('input'):
        # real images
        x = tf.placeholder(tf.float32, [None, image_px], name='x-input')
        # real lables
        y_ = tf.placeholder(tf.float32, [None, y_len], name='y-input')

    with tf.name_scope('input_reshape'):
        images_shaped_input = tf.reshape(x, [-1, 100, 60, 1])
        tf.summary.image('input', images_shaped_input, 10)

    W = tf.Variable(tf.zeros([image_px, y_len]))
    b = tf.Variable(tf.zeros([y_len]))
    y = tf.matmul(x, W) + b

    # forword prop
    predict = tf.argmax(y, axis=1)

    # Define loss and optimizer
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    with tf.Session() as sess:

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test')

        tf.global_variables_initializer().run()

        # Train
        saver = tf.train.Saver()
        for i in range(10000):
            batch_xs, batch_ys = train_data.next_batch(1000)

            if i % 100 == 0:
                # Test trained model
                summary, r = sess.run([merged, accuracy],
                                      feed_dict={x: test_data.images, y_: test_data.labels})
                test_writer.add_summary(summary, i)
                print 'step = %s, accuracy = %.2f%%' % (i, r * 100)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summar, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys},
                                     options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                saver.save(sess, log_dir+'/model.ckpt', i)

                a = sess.run(predict, feed_dict={x: test_data.images[-10:]})
                print 'predict: %s' % a
                print 'expect : %s ' % np.argmax(test_data.labels[-10:], axis=1)
            else:
                summar, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
                train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='images/one-char',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
