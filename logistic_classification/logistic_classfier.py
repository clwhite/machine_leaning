import tensorflow as tf
import np


def start():
  """
  filename_queue = tf.train.string_input_producer(['data-03-diabetes.csv'],
                                                  shuffle=False,
                                                  name='filename_queue')
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)

  record_default = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
  xy = tf.decode_csv(value, record_defaults=record_default)

  train_x_batch, train_y_batch = \
      tf.train.batch([xy[0:-1], xy[-1:]], batch_size=1)
  """

  xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
  x_data = xy[:, 0:-1]
  y_data = xy[:, [-1]]
  print "X_data:%s\nY_data:%s" % (x_data, y_data)

  X = tf.placeholder(tf.float32, shape=[None, 8])
  Y = tf.placeholder(tf.float32, shape=[None, 1])

  W = tf.Variable(tf.random_normal([8, 1]), name='weight')
  b = tf.Variable(tf.random_normal([1]), name='bias')

  hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

  cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train = optimizer.minimize(cost)

  # if hy > 0.5 = 1 else 0
  predicted = tf.cast(hypothesis > 0.65, dtype=tf.float32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # coord = tf.train.Coordinator()
  # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  print "start step"

  for step in range(10001):
    # x_data, y_data = sess.run([train_x_batch, train_y_batch])
    feed = {X: x_data, Y: y_data}
    sess.run(train, feed_dict=feed)
    cost_val = sess.run(cost, feed_dict=feed)
    if step % 200 == 0:
      print "step %s, cost:%s" % \
          (step, cost_val)

  # Accuracy report
  # x_data, y_data = sess.run([train_x_batch, train_y_batch])
  feed = {X: x_data, Y: y_data}
  h, c, a = sess.run([hypothesis, predicted, accuracy],
                     feed_dict=feed)
  print "hy:%s, correct:%s, accurcy:%s" % (h, c, a)
  # coord.request_stop()
  # coord.join(threads)


if __name__ == '__main__':
  start()
