# test for learning the detail code

import tensorflow as tf


# ------------- triplet-loss.py _pairwise_distances() -------------------

# x = tf.constant([(0,1.0,0),(1.0,2.0,2.0)])
# z = tf.matmul(x, tf.transpose(x))
# square_norm = tf.diag_part(z)
# sn1 = tf.expand_dims(square_norm, 1)
# sn2 = tf.expand_dims(square_norm, 0)
# distances = sn1 - 2.0 * z + sn2
#
# sess = tf.Session()
# # print("x=", sess.run(x))
# # print("tf.transpose(x)=", sess.run(tf.transpose(x)))
# # print("z=", sess.run(z))
# print("square_norm=", sess.run(square_norm))
# print("sn1=", sess.run(sn1))
# print("sn2=", sess.run(sn2))
# print("sn1 - 2.0 * z=", sess.run(sn1 - 2.0 * z))
# print("distances=", sess.run(distances))

# ------------- triplet-loss.py _pairwise_distances() -------------------



# ------------- triplet-loss.py _get_triplet_mask() -------------------

# # Check that i, j and k are distinct
# labels = [0, 1, 1]
# # Check that i and j are distinct
# indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
# indices_not_equal = tf.logical_not(indices_equal)
#
# # Check if labels[i] == labels[j]
# # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
# labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
#
# # Combine the two masks
# mask = tf.logical_and(indices_not_equal, labels_equal)
# mask = tf.to_float(mask)
#
# x = tf.constant([(0,1.0),(1.0,2.0),(5.0,5.0)])
# z = tf.matmul(x, tf.transpose(x))
# square_norm = tf.diag_part(z)
# sn1 = tf.expand_dims(square_norm, 1)
# sn2 = tf.expand_dims(square_norm, 0)
# distances = sn1 - 2.0 * z + sn2
#
# anchor_positive_dist = tf.multiply(mask, distances)
# sess = tf.Session()
# print("mask=", sess.run(mask))
# print("distances=", sess.run(distances))
# print("anchor_positive_dist=", sess.run(anchor_positive_dist))
#
# hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
# print("hardest_positive_dist=", sess.run(hardest_positive_dist))

# ------------- triplet-loss.py _get_triplet_mask() -------------------