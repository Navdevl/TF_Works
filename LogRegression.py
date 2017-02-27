import tensorflow as tf 
import numpy as np 
from sklearn.preprocessing import minmax_scale

weights = tf.Variable([0.1, 0.1])
xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)

x_data = np.array([[30, 80], [40, 60], [25, 70], [45, 50], [65, 80], [20, 15]])
y_data = np.array([0,1,0,1,1,0])

# x_data = np.array([65,50])
# y_data = np.array([1])

z = tf.reduce_sum(weights * xs, 1)
p = tf.exp(z) / (1 + tf.exp(z))
log = (ys * tf.log(p)) + ((1 - ys) * tf.log(1-p))
logloss =  log / -6
# logloss = - tf.reduce_sum(((ys * tf.log(p) ) + ( (1 - ys) * tf.log(1 - p)))) / 6
optimizer = tf.train.GradientDescentOptimizer(0.5)
minimize = optimizer.minimize(logloss)


sess = tf.Session()
sess.run(tf.initialize_all_variables())
xn_data  = minmax_scale(x_data)
# print xn_data
for i in range(10000):
	if i%1000 == 0 :
		print sess.run(tf.reduce_sum(sess.run([logloss], {xs: xn_data, ys: y_data})[0]))
	sess.run(minimize, {xs: xn_data, ys: y_data})

def minmax(x):
	min_a = 20
	max_a = 65
	min_b = 15
	max_b = 80
	x[0] = ( x[0] - min_a ) / (max_a - min_a)
	x[1] = ( x[1] - min_b ) / (max_b - min_b)
	print x
	return x


x = sess.run(z, {xs: [minmax([40.0,80.0])] })
print x


