import tensorflow as tf

#thank you to https://analysiscenter.github.io/dataset/_modules/dataset/models/tf/layers/core.html#alpha_dropout
#TODO: cite properly
def alpha_dropout(inputs, rate=0.5, noise_shape=None, seed=None, training=False, name=None):
    """ Alpha dropout layer

    Alpha Dropout is a dropout that maintains the self-normalizing property.
    For an input with zero mean and unit standard deviation, the output of Alpha Dropout maintains
    the original mean and standard deviation of the input.

    Klambauer G. et al "`Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_"
    """
    def _dropped_inputs():
        return tf.contrib.nn.alpha_dropout(inputs, 1-rate, noise_shape=noise_shape, seed=seed)
    return tf.cond(training, _dropped_inputs, lambda: tf.identity(inputs), name=name)

if __name__ == "__main__":
    pass
