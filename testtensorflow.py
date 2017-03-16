import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x=np.float32(np.random.rand(2,100))
w=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
b=tf.Variable(tf.zeros([1]))

y=tf.matmul(w,x)+b

y_1=np.dot([0.1,0.2],x)+0.3


cross=tf.reduce_mean(tf.square(y-y_1))


train=tf.train.GradientDescentOptimizer(0.5).minimize(cross)

init=tf.initialize_all_variables();
sess=tf.Session()
sess.run(init)


for step in range(0,201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(w),sess.run(b))
