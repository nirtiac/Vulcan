import numpy as np

import tensorflow as tf

from vulcanai.net import Network

from vulcanai.model_tests import run_test

from vulcanai.mnist_loader import load_fashion_mnist

from sklearn.utils import shuffle


train_images, train_labels, test_images, test_labels  = load_fashion_mnist()

print train_images.size
train_labels = train_labels.astype(np.int64)
test_labels = test_labels.astype(np.int64)

train_images, train_labels = shuffle(train_images, train_labels, random_state=0)


train_images_dict = {"MNISTIMAGE": train_images} #dictionary of features
test_images_dict = {"MNISTIMAGE": test_images}

#where we have one feature, an image. this would normally be more
#TODO: if we leave this here then we can create more complex columns from features...
#https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
feature_columns = []
for key in train_images_dict.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key, shape=train_images_dict[key].shape[1:]))

#TODO: will need to be dealt with differently when stitching networks.
input_var = None
y = "Something" #TODO: just a placeholder, need to figure out what to do with it

model_path = "train_mnist_test/"

network_dense_config = {
    'mode': 'dense',
    'units': [512, 512, 512],
    'dropouts': [0.2, 0.2, 0.2],
}

dense_net = Network(
    name='3_dense_test',
    dimensions=[None] + list(train_images.shape[1:]),
    input_var=input_var,
    y=y,
    config=network_dense_config,
    feature_columns = feature_columns,
    model_path = model_path,
    input_network=None,
    num_classes=10,
    activation='rectify',
    pred_activation='softmax',
    optimizer='adam')

# # Use to load model from disk
# # dense_net = Network.load_model('models/20170704194033_3_dense_test.network')
dense_net.train(
    epochs=10,
    train_x=train_images_dict,
    train_y=train_labels,
    val_x=test_images_dict,
    val_y=test_labels,
    batch_ratio=0.05,
    plot=True
)

dense_net.save_record()

run_test(dense_net, test_x=eval_data, test_y=eval_labels)
dense_net.save_model()
