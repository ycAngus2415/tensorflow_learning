import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plot

class RNNclass(object):
    def __init__(self, input_dim, output_dim, hiden_dim, deep=1):
        self.deep = deep
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiden_dim = hiden_dim
        self.sess = tf.Session()
    def forward(self, x, label):
        self.time_size = np.shape(x)[0]
        s= []
        for i in range(self.time_size):
             s.append(tf.zeros((self.hiden_dim), dtype='float32'))
        o = []
        for i in range(self.time_size):
            o.append(tf.zeros(self.output_dim, dtype='float32'))
        self.U = tf.Variable(tf.random_normal((self.hiden_dim, self.hiden_dim)), dtype='float32')
        self.W = tf.Variable(tf.random_uniform(shape=(self.input_dim, self.hiden_dim)), dtype='float32')
        self.O = tf.Variable(tf.random_normal((self.hiden_dim, self.output_dim)), dtype='float32')
        for t in np.arange(self.time_size):
            s[t] = tf.nn.relu(self.W * x[t] + tf.multiply(self.U, s[t-1]))
            o[t] = tf.nn.softmax(tf.multiply(self.O, s[t]))
        optimizer =tf.train.AdamOptimizer(0.01)
        loss = - tf.reduce_mean(label * tf.log(o))
        train = optimizer.minimize(loss)
        self.sess.run(tf.global_variables_initializer())
        for _ in range(10):
            self.sess.run(train)
            print(self.sess.run(loss))
        self.sess.close()

x = np.linspace(0, np.pi, 10)
y = np.sin(x)

rnn = RNNclass(1, 1, 1)
rnn.forward(x, y)
