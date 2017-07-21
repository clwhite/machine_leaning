import tensorflow as tf


def start():
  x_tranin = [1, 2, 3]
  y_train = [1, 2, 3]

  W = tf.Variable(2., name='weight')
  b = tf.Variable(5., name='bias')

  hypothesis = x_tranin * W + b

  # the square should be applied to calculate differences between
  # hypothesis - y_train, if there is no square, - value will be applied
  cost = tf.reduce_mean(tf.square(hypothesis - y_train))

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train = optimizer.minimize(cost)

  sess = tf.Session()
  sess.run(tf.initialize_all_variables())

  for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
      print(step, sess.run(cost), sess.run(W), sess.run(b))


if __name__ == '__main__':
  start()
