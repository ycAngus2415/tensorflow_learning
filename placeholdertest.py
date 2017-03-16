import tensorflow as tf


ss=tf.Session()
f1=tf.placeholder(tf.int32)
f2=tf.placeholder(tf.int32)
mu=tf.mul(f1,f2)
print(ss.run([mu],feed_dict={f1:[7],f2:[8]}))