import tensorflow as tf
import logging
import sys
import inspect


def test_constant():
  # test hello
  with tf.Session() as sess:
    hello = tf.constant("Hello")
    print("Hello:%s" % sess.run(hello))
    zeros = tf.zeros([1], dtype=tf.float32)
    print("zeros[1]:%s" % sess.run(zeros))
    ones = tf.ones([2, 3], dtype=tf.float32)
    print("ones[2, 3]:%s" % sess.run(ones))
    one_to_zero = tf.zeros_like(ones)
    print("ones_to_zero[2,3]:%s" % sess.run(one_to_zero))
    fill = tf.fill([2, 3], 9)
    print("fill [2,3]:%s" % sess.run(fill))
    lins = tf.linspace(0.0, 10.0, 20)
    print("linspace(0, 10, 20):%s" % sess.run(lins))


def test_random():
  with tf.Session() as sess:
    norm = tf.random_normal([2, 3], mean=-1, stddev=4)

    # Shuffle the first dimension of a tensor
    c = tf.constant([[1, 2], [3, 4], [5, 6]])
    shuff = tf.random_shuffle(c)
    print("Shuff of [1,2], [3,4], [5,6]")
    print(sess.run(shuff))

    # Each time we run these ops, different results are generated
    sess = tf.Session()
    print("Random normal double times")
    print(sess.run(norm))
    print(sess.run(norm))

    # Set an op-level seed to generate repeatable sequences across sessions.
    norm = tf.random_normal([2, 3], seed=1234)
    sess = tf.Session()
    print("Random normal with seed 1234")
    print(sess.run(norm))
    print(sess.run(norm))
    print("Random normal with seed with new session")
    sess = tf.Session()
    print(sess.run(norm))
    print(sess.run(norm))


def test_variable():
  # test x + 5
  x = tf.constant(35, name='x')
  y = tf.Variable(x + 5, name='y')

  model = tf.global_variables_initializer()

  # global_variables show all defined Variable previously
  print("variables after init:%s" % tf.global_variables())
  with tf.Session() as sess:
    sess.run(model)
    print sess.run(y)


def test_reduce_mean():
  x = tf.constant([[1., 2.], [3., 4.]])
  # mean for 1 + 2 + 3 + 4 / 4
  m1 = tf.reduce_mean(x)
  # mean for row [(1 + 2 / 2), (3 + 4 / 2)]
  m2 = tf.reduce_mean(x, 1)
  # mean for column [(1 + 3 / 2), (2 + 4 / 2)]
  m3 = tf.reduce_mean(x, 0)

  with tf.Session() as ss:
    print "reduce [[1,2], [3,4]] total:%s" % ss.run(m1)
    print "reduce [[1,2], [3,4]] row:%s" % ss.run(m2)
    print "reduce [[1,2], [3,4]] colum:%s" % ss.run(m3)


def test_assign():
  state = tf.Variable(0, name='counter')

  one = tf.constant(1)
  new_value = tf.add(state, one)
  # assign new_value to state variable
  update = tf.assign(state, new_value)

  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)
    print("first state:%s", sess.run(state))

    for _ in range(3):
      sess.run(update)
      print("new value in %s", sess.run(new_value))


def test_feed():
  in1 = tf.placeholder(tf.float32)
  in2 = tf.placeholder(tf.float32)
  output = in1 * in2

  in3 = tf.placeholder(tf.float32)
  y = in3 * 2

  with tf.Session() as sess:
    print(sess.run(output, feed_dict={in1: [3], in2: [2]}))
    result = sess.run(y, feed_dict={in3: [1, 2, 3]})
    # result array is numpy.ndarray
    print("run result:%s (type:%s)" % (result, type(result)))


def test_read_file():
  filename_queue = tf.train.string_input_producer(["data-01-test-score.csv"],
                                                  shuffle=False,
                                                  name='filename_queue')

  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)

  record_defaults = [[1.], [1.], [1.], [1.], [1.]]
  col1, col2, col3, total = tf.decode_csv(value,
                                          record_defaults=record_defaults)
  features = tf.stack([col1, col2, col3])

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1200):
      exam, result = sess.run([features, total])
      print "exam:%s, result:%s" % sess.run(exam, result)

    coord.request_stop()
    coord.join(thread)


if __name__ == '__main__':
  funcs = []
  for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isfunction(obj) and name.startswith('test'):
      funcs.append(obj)
  logging.info("test functions:%s", funcs)
  for f in funcs:
    logging.info("###############Test-%s", f.__name__)
    f()
