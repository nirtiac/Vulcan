"""Contains activation functions and gradient descent optimizers."""
import tensorflow as tf #TODO: import less

activations = {
    "sigmoid": tf.nn.sigmoid,
    "softmax": tf.nn.softmax,
    "rectify": tf.nn.relu, #TODO: this is it right?
    "selu": tf.nn.selu
}

#TODO: kinda dumb
optimizers = {
    "sgd": "sgd",
    "adam": "adam"
}
