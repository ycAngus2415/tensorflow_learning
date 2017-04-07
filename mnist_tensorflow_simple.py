import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
x = tf.placeholder("float", [None, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.placeholder("float", [None, 10])

loss = tf.reduce_mean(-y_*tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_: batch_y})
    print(sess.run(loss, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
