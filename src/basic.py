import numpy as np
import tensorflow as tf

print ("PACKAGES LOADED")


sess = tf.Session()
print ("Open Session")

def print_tf(x):
    print("Type is\n %s" % (type(x)))
    print("Value is\n %s" % (x))
hello = tf.constant("Hello. Its Me")
print_tf(hello)

weight = tf.Variable(tf.random_normal([5, 2], stddev=0.1))
print_tf(weight)

init = tf.initialize_all_variables()
sess.run(init)

weight_out = sess.run(weight)
print_tf(weight_out)

x = tf.placeholder(tf.float32, [None, 5])
print_tf(x)

oper = tf.matmul(x, weight)
print_tf(oper)

data = np.random.rand(2, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)
