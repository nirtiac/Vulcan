import numpy as np

import tensorflow as tf

from vulcanai.net import Network

from vulcanai.utils import get_one_hot

from vulcanai import mnist_loader

from vulcanai.model_tests import run_test

(train_images, train_labels, test_images, test_labels) = mnist_loader.load_fashion_mnist()

#TODO: how are feature names passed??? would be nice to do it as feature_columns as seen here:
    # Feature columns describe how to use the input.

    #my_feature_columns = []
    #for key in train_x.keys():
    #    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

train_labels = get_one_hot(train_labels)

#TODO: will need to be dealt with differently when stitching networks.
input_var = None
y = "Something" #TODO: just a placeholder, need to figure out what to do with it

network_dense_config = {
    'mode': 'dense',
    'units': [512],
    'dropouts': [0.2],
}

dense_net = Network(
    name='3_dense_test',
    dimensions=[None] + list(train_images.shape[1:]),
    input_var=input_var,
    y=y,
    config=network_dense_config,
    input_network=None,
    num_classes=10,
    activation='rectify',
    pred_activation='softmax',
    optimizer='adam')

# # Use to load model from disk
# # dense_net = Network.load_model('models/20170704194033_3_dense_test.network')
dense_net.train(
    epochs=2,
    train_x=train_images[:50000],
    train_y=train_labels[:50000],
    val_x=train_images[50000:60000],
    val_y=train_labels[50000:60000],
    batch_ratio=0.05,
    plot=True

)

dense_net.save_record()

run_test(dense_net, test_x=train_images[50000:60000], test_y=train_labels[50000:60000])
dense_net.save_model()
