import tensorflow as tf

state=tf.Variable(0)
one=tf.constant(1);
ob=tf.add(state,one)
update=tf.assign(state,ob)


sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init  )
print(sess.run(state))


for i in range(3):
    result=sess.run([update,one])
    print(result)
