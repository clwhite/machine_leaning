import tensorflow as tf
import np


def normalize(data, indexs):
  for index in indexs:
    mn = 0
    mx = 0
    for d in data:
      if d[index] < mn:
        mn = d[index]
      if d[index] > mx:
        mx = d[index]
    for d in data:
      d[index] = (d[index] - mn) / (mx - mn)
  return data


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
  fq = tf.train.string_input_producer(['bank-traning.csv'],
                                      shuffle=False,
                                      name='filename_queue')

  reader = tf.TextLineReader()

  key, value = reader.read(fq)

  record_default = [[0.], [0.], [0.], [0.], [0.],
                    [0.], [0.], [0.], [0.], [0.],
                    [0.], [0.], [0.], [0.], [0.],
                    [0.], [0.]]
  xy = tf.decode_csv(value, record_defaults=record_default)

  train_x_batch, train_y_batch = \
      tf.train.batch([xy[0:-1], xy[-1:]], batch_size=1000)

  """
  xy = np.loadtxt('bank-training.csv', delimiter=',', dtype=np.float32)
  x_data = xy[:, 0:-1]
  y_data = xy[:, [-1]]

  x_data = normalize(x_data, [0, 1, 2, 3, 4, 5,
                              6, 7, 8, 9, 10, 11,
                              12, 13, 14, 15])
  print x_data

  X = tf.placeholder(tf.float32, shape=[None, 16])
  Y = tf.placeholder(tf.float32, shape=[None, 1])

  W = tf.Variable(tf.random_normal([16, 1]), name='weight')
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

  c = None
  for step in range(5001):
    # x_data, y_data = sess.run([train_x_batch, train_y_batch])
    feed = {X: x_data, Y: y_data}
    sess.run(train, feed_dict=feed)
    cost_val = sess.run(cost, feed_dict=feed)
    if step % 200 == 0:
      c = cost_val
      print "step %s, cost:%s" % \
          (step, cost_val)

  # Accuracy report
  t_xy = np.loadtxt('bank-test.csv', delimiter=',', dtype=np.float32)
  # t_xy = xy
  tx_data = t_xy[:, 0:-1]
  tx_data = normalize(tx_data, [0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10, 11,
                                12, 13, 14, 15])
  ty_data = t_xy[:, [-1]]
  # x_data, y_data = sess.run([train_x_batch, train_y_batch])
  feed = {X: tx_data, Y: ty_data}
  h, c, a = sess.run([hypothesis, predicted, accuracy],
                     feed_dict=feed)
  print "hy:%s, correct:%s, accurcy:%s" % (h, c, a)
  # coord.request_stop()
  # coord.join(threads)
  return a, c


if __name__ == '__main__':
  result = []
  for i in range(10):
    a, c = start()
    result.append((a, c))
  for a, c in result:
    print "acuracy:%s, cost:%s" % (a, c)
