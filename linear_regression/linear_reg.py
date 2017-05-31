import tensorflow as tf
import matplotlib.pyplot as plt

import logging
import sys


def _setup_logging():
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)

  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  root.addHandler(ch)


def start():
  _setup_logging()
  x_tranin = [1, 2, 3]
  y_train = [1, 3, 3]

  logging.info("Start tf!!")

  W = tf.Variable(tf.random_normal([1]), name='weight')
  b = tf.Variable(tf.random_normal([1]), name='bias')
  logging.info("W:%s, b:%s", W, b)

  # Our hypothesis XW + b
  hypothesis = x_tranin * W + b
  logging.info("Hypothesis:%s", hypothesis)

  # cost/loss function
  cost = tf.reduce_mean(tf.square(hypothesis - y_train))
  logging.info("cost:%s", cost)

  # Minimize
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train = optimizer.minimize(cost)
  logging.info("train:%s", train)

  # Launch the graph in a session.
  sess = tf.Session()
  # Initializes global variables in the graph.
  sess.run(tf.initialize_all_variables())

  # Fit the line
  for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
      print(step, sess.run(cost), sess.run(W), sess.run(b))


if __name__ == '__main__':
  start()
