import tensorflow as tf


def start():
  filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'],
                                                  shuffle=False,
                                                  name='filename_queue')
  print "file name queue:%s" % filename_queue

  reader = tf.TextLineReader()
  print "reader:%s" % reader

  key, value = reader.read(filename_queue)
  print "key:%s, value:%s" % (key, value)

  record_default = [[0.], [0.], [0.], [0.]]
  xy = tf.decode_csv(value, record_defaults=record_default)
  print "xy:%s" % xy

  train_x_batch, train_y_batch = \
      tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
  print 'x batch:%s\ny batch:%s' % (train_x_batch, train_y_batch)

  X = tf.placeholder(tf.float32, shape=[None, 3])
  Y = tf.placeholder(tf.float32, shape=[None, 1])

  W = tf.Variable(tf.ones([3, 1]), name='weight')
  b = tf.Variable(tf.ones([1]), name='bias')

  hypothesis = tf.matmul(X, W) + b

  cost = tf.reduce_mean(tf.square(hypothesis - Y))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
  train = optimizer.minimize(cost)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  print "start step"

  for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
      print "step %s, cost:%s W:%s, b:%s" % \
          (step, cost_val, sess.run(W), sess.run(b))

  print "Start evaluation"
  x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
  print "Prediction\n%s" % (y_batch)
  result = sess.run(hypothesis, feed_dict={X: x_batch})
  print "Result\n%s" % (result)

  coord.request_stop()
  coord.join(threads)

if __name__ == '__main__':
  start()
