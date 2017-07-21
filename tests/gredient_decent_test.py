import tensorflow as tf
import logging
import sys
import inspect


def test_gradient_decent():
  x = tf.Variable(10., name='weight')
  b = tf.Variable(5., name='bias')

  cost = tf.square(x - b) 
  # Find lowest point of x and b of square(x - b)
  train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
  with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())
    for step in range(2001):
      ss.run(train)
      if step % 20 == 0:
        print(step, ss.run(cost), ss.run(W), ss.run(b))


if __name__ == '__main__':
  funcs = []
  for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isfunction(obj) and name.startswith('test'):
      funcs.append(obj)
  logging.info("test functions:%s", funcs)
  for f in funcs:
    logging.info("###############Test-%s", f.__name__)
    f()
