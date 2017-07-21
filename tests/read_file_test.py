import tensorflow as tf
import logging
import sys
import inspect


def test_read_file():
  filename_queue = tf.train.string_input_producer(["data-01-test-score.csv"],
                                                  shuffle=False,
                                                  name='filename_queue')

  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)

  record_defaults = [[1.], [1.], [1.], [1.]]
  col1, col2, col3, total = tf.decode_csv(value,
                                          record_defaults=record_defaults)
  x_batch, y_batch = tf.train.batch([[col1, col2, col3], total], batch_size=5)

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1200):
      x, y = sess.run([x_batch, y_batch])
      print "exam:%s, result:%s" % (x, y)

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
