"""Train the model"""

import argparse
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.cluster import KMeans
from sklearn import metrics

import model.mnist_dataset as mnist_dataset
from model.utils import Params
from model.input_fn import train_embedding_fn
from model.input_fn import train_input_fn
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/mnist',
                    help="Directory containing the dataset")
parser.add_argument('--sprite_filename', default='experiments/mnist_10k_sprite.png',
                    help="Sprite image for the projector")


def __get_embedding__():

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

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





    # predictions = res["predictions"]
    # labels = res["labels"]
    #
    # print("predictions=", predictions, "labels=", labels)
    # for i, p in enumerate(predictions):
    #     print("predictions=", p)

    # embeddings = np.zeros((params.train_size, params.embedding_size))
    # print("embeddings.size=", embeddings.shape)
    # for i, p in enumerate(predictions):  # i:enumerate_id, p:{'embeddings':array(64)}
    #     if i >= params.train_size:  # don't know why it can reach params.train_size
    #         break
    #     embeddings[i] = p['embeddings']
    #
    # tf.logging.info("Embeddings shape: {}".format(embeddings.shape))  # (50000, 64)
    #
    # # use k-means to create centroid for knn
    # # run k-means for each class otherwise you can't figure out knn's nearest neighbour's lable
    #
    # kmeans = KMeans(n_clusters=15, random_state=0).fit(embeddings)
    # print(kmeans.cluster_centers_)
    # print(kmeans.cluster_centers_.shape)


    # cluster_centers = [ [] for _ in range(10) ]



    # print("k-means score=", metrics.calinski_harabaz_score(embeddings, y_pred))

    # merge the centroid's embedding with their labels(or separately save them)


    # save as npy file

def __get_label__():

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    with tf.Session() as sess:
        # Obtain the test labels
        dataset = mnist_dataset.train(args.data_dir)
        dataset = dataset.map(lambda img, lab: lab)
        dataset = dataset.batch(params.train_size)
        labels_tensor = dataset.make_one_shot_iterator().get_next()
        labels = sess.run(labels_tensor)

    return labels

if __name__ == '__main__':

    embeddings = __get_embedding__()
    labels = __get_label__()
    # print("embeddings.shape=", embeddings.shape)
    # print("labels.shape=", labels.shape)

    # now we have embeddings and labels of training set in the same order
    
