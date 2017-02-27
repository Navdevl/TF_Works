import tensorflow as tf 
import numpy as np 

weight = tf.Variable([1.0])
bias = tf.Variable([0.0])


xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)

x_data = np.array([1, 2])
y_data = np.array([2, 4])

output = ( weight * xs ) 
diff =  output - ys
cost = tf.square(diff)
grad1 = tf.gradients(cost/2, weight)
grad = tf.reduce_sum(((weight * xs) - ys ) * xs)
optimized_weight = weight - 0.1 * grad
reduced_cost = tf.reduce_mean(cost) / 2
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# sess.run([weight], {xs: x_data, ys: y_data})
for i in range(10):
	print ''
	print 'Step: ', i
	print 'Gradient: ', sess.run([grad1, grad], {xs: x_data, ys: y_data})
	print 'Weight: ' , sess.run([weight])[0][0]
	print 'Bias: ', sess.run([bias])[0][0]
	print 'Loss: ', sess.run([reduced_cost], {xs: x_data, ys: y_data})[0]
	print 'Optimized Weight: ', sess.run([optimized_weight], {xs: x_data, ys: y_data})[0]
	sess.run(tf.assign(weight, sess.run([optimized_weight], {xs: x_data, ys: y_data})[0]))
	# O_weight =  sess.run([optimized_weight], {xs: x_data, ys: y_data})[0]



