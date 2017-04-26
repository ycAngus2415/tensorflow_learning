import tensorflow as tf

merged_summary_op = tf.summary()
summary_writer = tf.train.SummarySaverHook()
with tf.name_scope('hidden') as scope:
    a = tf.constant(5, name='alpha')
    w = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0), name='weights')
    b = tf.Variable(tf.zeros([1], name='biases'))
