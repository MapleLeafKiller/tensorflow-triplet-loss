# restore from ckpt file

import tensorflow as tf

tf.reset_default_graph()

# Create some variables.
images = tf.get_variable("features", shape=[50000, 28, 28, 1])
labels = tf.get_variable("labels", shape=[50000])


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "../experiments/batch_hard/model.ckpt-0")
  print("Model restored.")
  # Check the values of the variables
  print("images:", images.eval())
  print("labels:", labels.eval())
