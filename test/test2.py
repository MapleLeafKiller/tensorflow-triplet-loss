# test function _pairwise_distances()

import numpy as np
import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
from model.triplet_loss import _pairwise_distances
from model.triplet_loss import _get_triplet_mask
from model.triplet_loss import _get_anchor_positive_triplet_mask
from model.triplet_loss import _get_anchor_negative_triplet_mask

#
# def pairwise_distance_np(feature, squared=False):
#     """Computes the pairwise distance matrix in numpy.
#     Args:
#         feature: 2-D numpy array of size [number of data, feature dimension]
#         squared: Boolean. If true, output is the pairwise squared euclidean
#                  distance matrix; else, output is the pairwise euclidean distance matrix.
#     Returns:
#         pairwise_distances: 2-D numpy array of size
#                             [number of data, number of data].
#     """
#     triu = np.triu_indices(feature.shape[0], 1)
#     upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
#     if squared:
#         upper_tri_pdists **= 2.
#     num_data = feature.shape[0]
#     pairwise_distances = np.zeros((num_data, num_data))
#     pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
#     # Make symmetrical.
#     pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
#             pairwise_distances.diagonal())
#     return pairwise_distances
#
#
# """Test the pairwise distances function."""
# num_data = 64
# feat_dim = 6
#
# embeddings = np.random.randn(num_data, feat_dim).astype(np.float32)
# embeddings[1] = embeddings[0]  # to get distance 0
#
# with tf.Session() as sess:
#     # for squared in [True, False]:
#     #     res_np = pairwise_distance_np(embeddings, squared=squared)
#     #     res_tf = sess.run(_pairwise_distances(embeddings, squared=squared))
#     #     assert np.allclose(res_np, res_tf)
#     res_tf = sess.run(_pairwise_distances(embeddings, squared=False))
#     print(res_tf.shape)

"""Test function _get_anchor_positive_triplet_mask."""
num_data = 6
num_classes = 10

labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

mask_np = np.zeros((num_data, num_data))
for i in range(num_data):
    for j in range(num_data):
        distinct = (i != j)
        valid = labels[i] == labels[j]
        mask_np[i, j] = (distinct and valid)

mask_tf = _get_anchor_positive_triplet_mask(labels)

assert mask_tf.shape == [64,64]