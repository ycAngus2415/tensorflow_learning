import tensorflow as tf

x = tf.constant(1.0, name="input")
w = tf.Variable(8.0, name="weight")
y_1 = tf.multiply(w, x, name="output")
w_1 = tf.Variable(7.0, name='weight1')
y = tf.multiply(w_1, y_1, name='output1')
y_ = tf.constant(0.0)
loss = (y - y_)**2
optimizer = tf.train.GradientDescentOptimizer(0.01)
grads_and_vars = optimizer.compute_gradients(loss)
train = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
summ = tf.summary.FileWriter('log')
summ_y = tf.summary.scalar('outpu1', y)
for i in range(100):
    summ_str = sess.run(summ_y)
    summ.add_summary(summ_str, i)
    sess.run(train)
    print(sess.run(y))
