"""Calculate test set' accuracy by:
   First generate (embeddings,labels) of training set
   Then perform k-means for each class and get centroids_per_class, save as npy file
   When predicting, generate embedding of test set and perform knn to predict labels
"""

import argparse
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


import model.mnist_dataset as mnist_dataset
from model.utils import Params
from model.input_fn import train_embedding_fn
from model.input_fn import train_input_fn
from model.input_fn import test_input_fn
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/mnist',
                    help="Directory containing the dataset")
parser.add_argument('--sprite_filename', default='experiments/mnist_10k_sprite.png',
                    help="Sprite image for the projector")


# calculate embeddings of training set and perform k-means each class, save centroids' embeddings
def __get_centroids__(args, params):
    # get embeddings and labels of training set in the same order
    embeddings = __get_embedding__(args, params)
    labels = __get_label__(args, params)
    labels = np.array(labels)
    # print("embeddings.shape=", embeddings.shape)
    # print("labels.shape=", labels.shape)

    # perform k-means for each class(0~9)
    images_per_class = [[] for _ in range(params.num_labels)]
    for i in range(len(labels)):
        images_per_class[labels[i]].append(embeddings[i])

    kmeans = []
    n_clusters = 1000
    centroids = np.zeros(shape=[params.num_labels, n_clusters, params.embedding_size], dtype=np.float32)
    scores = 0
    for i in range(len(images_per_class)):
        kmean = KMeans(n_clusters=n_clusters, random_state=0).fit(images_per_class[i])
        kmeans.append(kmean)
        centroids[i] = kmean.cluster_centers_
        # print("centroids[", i, "]=", np.array(centroids[i]).shape)  # (1,n_clusters,64)
        y_pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(images_per_class[i])
        score = metrics.calinski_harabaz_score(images_per_class[i], y_pred)
        scores += score
    print("scores=", scores)

    # save the centroids as npy file, the label of centroid starts from 0 to num_labels-1, each n_clusters add 1
    path = os.path.join(args.model_dir, "centroids_embeddings.npy")
    np.save(path, centroids)


# get embeddings of training set in the right order(no shuffle)
def __get_embedding__(args, params):

    # Define the model
    tf.logging.info("-------Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Compute embeddings on the training set
    tf.logging.info("-------Compute embeddings on the training set,remember:No shuffle!!!")
    predictions = estimator.predict(lambda: train_embedding_fn(args.data_dir, params))

    embeddings = np.zeros((params.train_size, params.embedding_size))
    for i, p in enumerate(predictions):  # i:enumerate_id, p:{'embeddings':array(64)}
        if i >= params.train_size:  # don't know why it can reach params.train_size
            break
        embeddings[i] = p['embeddings']

    tf.logging.info("Embeddings shape: {}".format(embeddings.shape))  # (50000, 64)

    return embeddings


#  get labels of training set in the right order(no shuffle)
def __get_label__(args, params):

    with tf.Session() as sess:
        # Obtain the test labels
        dataset = mnist_dataset.train(args.data_dir)
        dataset = dataset.map(lambda img, lab: lab)
        dataset = dataset.batch(params.train_size)
        labels_tensor = dataset.make_one_shot_iterator().get_next()
        labels = sess.run(labels_tensor)

    return labels


# generate embedding of test set and perform knn to predict labels
def __predict__(args, params):
    path = os.path.join(args.model_dir, "centroids_embeddings.npy")
    centrois = np.load(path)
    n_clusters = centrois.shape[1]  # remember to get it before reshape
    # print("centrois.shape=", centrois.shape)
    centrois = centrois.reshape(-1, centrois.shape[2])  # flatten it for knn
    # print("centrois.shape=", centrois.shape)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Compute embeddings on the test set
    tf.logging.info("Predicting")
    predictions = estimator.predict(lambda: test_input_fn(args.data_dir, params))

    embeddings = np.zeros((params.eval_size, params.embedding_size))
    for i, p in enumerate(predictions):  # i:enumerate_id, p:{'embeddings':array(64)}
        embeddings[i] = p['embeddings']

    labels = np.zeros(shape=[params.num_labels * n_clusters], dtype=np.int)
    for i in range(1, params.num_labels):
        for j in range(n_clusters):
            labels[n_clusters*i+j] = i

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(centrois, labels)
    y_predicted = knn.predict(embeddings)

    with tf.Session() as sess:
        dataset = mnist_dataset.test(args.data_dir)
        dataset = dataset.map(lambda img, lab: lab)
        dataset = dataset.batch(params.eval_size)
        labels_tensor = dataset.make_one_shot_iterator().get_next()
        y_true = sess.run(labels_tensor)

    print("Accuracy: " + str(metrics.accuracy_score(y_true, y_predicted) * 100) + "%")


if __name__ == '__main__':

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # run only once
    # calculate embeddings of training set and perform k-means each class, save centroids' embeddings
    __get_centroids__(args, params)

    # calculate accuracy
    __predict__(args, params)
