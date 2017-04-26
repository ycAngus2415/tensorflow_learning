import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

#w = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
#init = tf.global_variables_initializer()
#
#y = tf.nn.softmax(tf.matmul(x, w) + b)
#
#cross_loss = -tf.reduce_mean(y_*tf.log(y))
#optimizer = tf.train.GradientDescentOptimizer(0.01)
#train = optimizer.minimize(cross_loss)
#
#sess = tf.Session()
#sess.run(init)
#for i in range(1000):
#    batch_x, batch_y = mnist.train.next_batch(50)
#    sess.run(train, feed_dict={x: batch_x, y_: batch_y})
#print(sess.run(cross_loss, {x: mnist.test.images, y_: mnist.test.labels}))
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuray = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(sess.run(accuray, {x: mnist.test.images, y_: mnist.test.labels}))
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)#w:height, width, in_channel, out_channel

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
#x:batch, height, widht, channel
def max_pool2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2,1], padding='SAME')
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool2(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool2(h_conv2)

w_f1 = weight_variable([7*7*64, 1024])
b_f1 = bias_variable([1024])
#全链接层
h_pool2_f = tf.reshape(h_pool2, [-1, 7*7*64])
h_f1 = tf.nn.relu(tf.matmul(h_pool2_f, w_f1) + b_f1)
#dropout 层
keep_prob = tf.placeholder("float")
h_f1_drop = tf.nn.dropout(h_f1, keep_prob)
#输出层
w_f2 = weight_variable([1024, 10])
b_f2 = bias_variable([10])

y = tf.nn.softmax(tf.matmul(h_f1_drop, w_f2) + b_f2)

loss = -tf.reduce_mean(y_*tf.log(y), name='loss')
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
init = tf.global_variables_initializer()
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
summary_writer = tf.summary.FileWriter('log')
summary_scalar = tf.summary.scalar('loss', loss)
with tf.Session() as sess:
    sess.run(init)
    t0 = time.clock()
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        summary_str = sess.run(summary_scalar, feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        if i % 100 ==0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print(time.clock() - t0)
            print(i,"accuracy:", train_accuracy)
        train.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('test accuracy %g'%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))
