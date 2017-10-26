import tensorflow as tf

def run()
	output = None
	x = tf.placeholder(tf.int32)

	with tf.Session() as sess:
		output = sess.run(x, feed_dict={x: 123})

	return output


import tensorflow as tf

x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x, y), tf.constant(1))

with tf.Session() as sess:
	output = sess.run(z)