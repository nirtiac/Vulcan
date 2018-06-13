"""Contains activation functions and gradient descent optimizers."""
import tensorflow as tf #TODO: import less

activations = {
    "sigmoid": tf.nn.sigmoid,
    "softmax": tf.nn.softmax,
    "rectify": tf.nn.rectify,
    "selu": tf.nn.selu
}

optimizers = {
    "sgd": sgd,
    "adam": adam
}
