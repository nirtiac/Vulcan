"""Contains the class for creating networks."""

import time
import os
import sys
import json
import numpy as np
import tensorflow as tf
from utils import get_timestamp
from ops import activations, optimizers
from selu import alpha_dropout

sys.setrecursionlimit(5000)

tf.logging.set_verbosity(tf.logging.INFO)


class Network(object):
    """Class to generate networks and train them."""

    def __init__(self, name, dimensions, input_var, y, config, feature_columns, model_path,
                 input_network=None, num_classes=None, activation='rectify',
                 pred_activation='softmax', optimizer='adam', stopping_rule='best_validation_error',
                 learning_rate=0.001):

        """
        Initialize network specified.

        Args:
            name: string of network name
            dimensions: the size of the input data matrix
            input_var: tf tensor representing input matrix
            y: tf tensor representing truth matrix
            config: Network configuration (as dict)
            feature_columns: a list of feature columns as defined by tensorflow
            input_network: None or a dictionary containing keys (network, layer).
                network: a Network object
                layer: an integer corresponding to the layer you want output
            num_classes: None or int. how many classes to predict
            activation:  activation function for hidden layers
            pred_activation: the classifying layer activation
            optimizer: which optimizer to use as the learning function
            learning_rate: the initial learning rate
        """
        self.name = name
        self.layers = []
        #self.cost = None
        #self.val_cost = None #TODO: figure out if you really need these.
        self.input_dimensions = dimensions #TODO: I think for non-numeric this needs to contain more info
        #TODO: confirm dimensions is what you think it is for input networks
        self.config = config
        self.learning_rate = learning_rate
        self.init_learning_rate = learning_rate
        self.stopping_rule = stopping_rule
        if not optimizers.get(optimizer, False):
            raise ValueError(
                'Invalid optimizer option: {}. '
                'Please choose from:'
                '{}'.format(optimizer, optimizers.keys()))
        if not activations.get(activation, False) or \
           not activations.get(pred_activation, False):
            raise ValueError(
                'Invalid activation option: {} and {}. '
                'Please choose from:'
                '{}'.format(activation, pred_activation, activations.keys()))
        self.activation = activation
        self.pred_activation = pred_activation
        self.optimizer = optimizer
        self.input_var = input_var
        self.y = y
        self.input_network = None  #TODO: undo this, deal with input networks
        self.input_params = None
        # if self.input_network is not None:
        #     if self.input_network.get('network', False) is not False and \
        #        self.input_network.get('layer', False) is not False and \
        #        self.input_network.get('get_params', None) is not None:
        #             pass  #TODO insert the tensorflow stuff here

        #     else:
        #         raise ValueError(
        #             'input_network for {} requires {{ network: type Network,'
        #             ' layer: type int, get_params: type bool}}. '
        #             'Only given keys: {}'.format(
        #                 self.name, self.input_network.keys()
        #             )
        #         )
        self.num_classes = num_classes

        self.feature_columns = feature_columns

        self.model_path = model_path

        model_function = self.make_model_function()

        self.classifier = tf.estimator.Estimator(model_fn=model_function, model_dir=self.model_path)

        #so this is like predictions?? confused.
        # self.output = theano.function(
        #     [i for i in [self.input_var] if i],
        #     lasagne.layers.get_output(self.network, deterministic=True))

        self.record = None
        self.tf_is_training = False
        try:
            self.timestamp
        except AttributeError:
            self.timestamp = get_timestamp()

    def make_model_function(self):

        def my_model(features, labels, mode):

            network = self.create_network(features, config=self.config, nonlinearity=activations[self.activation])

            predicted_classes = tf.argmax(network, 1)
            print predicted_classes
            predictions = {
                    'class_ids': predicted_classes,
                    'probabilities': tf.nn.softmax(network, name="softmax_tensor"),
                    'logits': network,
                }
            if mode == tf.estimator.ModeKeys.PREDICT:

                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            if self.num_classes is None or self.num_classes == 0:
                loss = tf.losses.mean_squared_error(labels=labels, logits=network)

            else:
                loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=network) #TODO: check sparse

            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=predicted_classes,
                                           name='acc_op')

            metrics = {'accuracy': accuracy}

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            assert mode == tf.estimator.ModeKeys.TRAIN

            #TODO: fix learning rate
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            else:
                ValueError("No optimizer found")

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            #sets the tensorboard scalar accuracy
            tf.summary.scalar('accuracy', accuracy[1])

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        return my_model

    def create_network(self, features, config, nonlinearity):
        """
        Abstract function to create any network given a config dict.

        Args:
            config: dict. the network configuration
            nonlinearity: string. the nonlinearity to add onto each layer

        returns a network.
        """
        import jsonschema
        import schemas
        mode = config.get('mode')

        if mode == 'dense':
            jsonschema.validate(config, schemas.dense_network)
            network = self.create_dense_network(
                features,
                units=config.get('units'),
                dropouts=config.get('dropouts'),
                nonlinearity=nonlinearity
            )

        elif mode == 'conv':
            jsonschema.validate(config, schemas.conv_network)
            network = self.create_conv_network(
                filters=config.get('filters'),
                filter_size=config.get('filter_size'),
                stride=config.get('stride'),
                pool_mode=config['pool'].get('mode'),
                pool_stride=config['pool'].get('stride'),
                nonlinearity=nonlinearity
            )
        else:
            raise ValueError('Mode {} not supported.'.format(mode))
        #TODO: check this is in the right place
        network = self.create_classification_layer(network, nonlinearity=activations[self.pred_activation])

        return network

    def create_classification_layer(self, network, nonlinearity):

        logits = tf.layers.dense(network, self.num_classes, activation=nonlinearity, name="classification_layer")

    def create_dense_network(self, features, units, dropouts, nonlinearity):
        """
        Generate a dense network.

        Args:
            features: diction
            units: The list of number of nodes to have at each layer
            dropouts: The list of dropout probabilities for each layer
            nonlinearity: Nonlinearity from Lasagne.nonlinearities

        Returns: the output of the network (linked up to all the layers)
        """

        if len(units) != len(dropouts):
            raise ValueError(
                "Cannot build network: units and dropouts don't correspond"
            )

        #doesn't have a name param
        network = tf.feature_column.input_layer(features, self.feature_columns)

        for i, (num_units, prob_dropout) in enumerate(zip(units, dropouts)):

            network = tf.layers.dense(network, units=num_units, activation=nonlinearity,name = 'dense_layer{}'.format(str(i)))

            if self.activation == "selu":
                network = alpha_dropout(network, rate=prob_dropout, training=self.tf_is_training, name = 'dropout_layer{}'.format(str(i)))
            else:
                network = tf.layers.dropout(network, rate=prob_dropout, training=self.tf_is_training, name = 'dropout_layer{}'.format(str(i)))

        return network


    def eval_input_fn(self, features, labels, batch_ratio):
        """An input function for evaluation or prediction"""
        features=dict(features)
        if labels is None:
            inputs = features
        else:
            inputs = (features, labels)

        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        batch_size = int(len(labels) * batch_ratio)

        if batch_size < 1:
            batch_size = len(labels)

        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        return dataset


    #TODO: make this flexible for on-demand dataset use
    #https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/blog_estimators_dataset.py
    def train_input_fn(self, features, labels, batch_ratio):
        """An input function for training"""
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)) #TODO: this will probably break

        batch_size = int(len(labels) * batch_ratio)

        if batch_size < 1:
            batch_size = len(labels)

        #TODO set shuffle param?
        dataset = dataset.shuffle(1000).batch(batch_size)

        return dataset

    def train(self, epochs, train_x, train_y, val_x, val_y,
              batch_ratio=0.1, plot=True, change_rate=None):
        """
        Train the network.

        Args:
            epochs: how many times to iterate over the training data
            train_x: the training data, a dictionary of features: tensors
            train_y: the training truth
            val_x: the validation data (should not be also in train_x)
            val_y: the validation truth (should not be also in train_y)
            batch_ratio: the percent (0-1) of how much data a batch should have
            plot: If True, plot performance during training
            change_rate: a function that updates learning rate (takes an alpha, returns an alpha)'

        """

        self.tf_is_training = True #for dropout #TODO: check that this works
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        if batch_ratio > 1:
            batch_ratio = 1
        batch_ratio = float(batch_ratio)

        self.record = dict(
            epoch=[],
            train_error=[],
            train_accuracy=[],
            validation_error=[],
            validation_accuracy=[]
        )

        if self.stopping_rule == 'best_validation_error':
            best_epoch = None
            best_error = float('inf')

        elif self.stopping_rule == 'best_validation_accuracy':
            best_epoch = None
            best_accuracy = 0.0

        #TODO: catch all the errors!!!
        #TODO: need to be able to update the learning rate. I think you need to make it a tensor object in your model fun optimizer PRIYA
        #https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer
        #https://github.com/tensorflow/tensorflow/issues/2198


        # in the future tf will hopefully have early stopping implemented
        # https://github.com/tensorflow/tensorflow/issues/18394

        #TODO: actually use this https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate
        try:

            for epoch in range(epochs):
                epoch_time = time.time()

                self.classifier.train(input_fn=lambda:self.train_input_fn(train_x, train_y, batch_ratio), hooks=[logging_hook])

                train_eval_result = self.classifier.evaluate(input_fn=lambda:self.eval_input_fn(train_x, train_y, batch_ratio))
                val_eval_result = self.classifier.evaluate(input_fn=lambda:self.eval_input_fn(val_x, val_y, batch_ratio))

                train_error = train_eval_result["loss"]
                train_accuracy = train_eval_result["accuracy"]
                validation_error = val_eval_result["loss"]
                validation_accuracy = val_eval_result['accuracy']

                if self.stopping_rule == 'best_validation_error' and validation_error < best_error:
                    best_epoch = epoch
                    best_error = validation_error

                elif self.stopping_rule == 'best_validation_accuracy' and validation_accuracy > best_accuracy:
                    best_epoch = epoch
                    best_accuracy = validation_accuracy

                self.record['epoch'].append(epoch)
                self.record['train_error'].append(train_error)
                self.record['train_accuracy'].append(train_accuracy)
                self.record['validation_error'].append(validation_error)
                self.record['validation_accuracy'].append(validation_accuracy)

                epoch_time_spent = time.time() - epoch_time

                print("\n\ttrain error: {:.6f} |"" train accuracy: {:.6f} in {:.2f}s".format(
                    float(train_error),
                    float(train_accuracy),
                    epoch_time_spent))

                print("\tvalid error: {:.6f} | valid accuracy: {:.6f} in {:.2f}s".format(
                    float(validation_error),
                    float(validation_accuracy),
                    epoch_time_spent))

                eta = epoch_time_spent * (epochs - epoch - 1)
                minute, second = divmod(eta, 60)
                hour, minute = divmod(minute, 60)
                print("\tEstimated time left: {}:{}:{} (h:m:s)\n".format(
                    int(hour),
                    int(minute),
                    int(second)))

        except KeyboardInterrupt:
            print("\n\n**********Training stopped prematurely.**********\n\n")

        finally:
            self.timestamp = get_timestamp()

            if self.stopping_rule == 'best_validation_error':
                print("STOPPING RULE: Rewinding to epoch {} which had the lowest validation error: {}\n".format(best_epoch, best_error))
                #self.__setstate__(best_state)

            elif self.stopping_rule == 'best_validation_accuracy':
                print("STOPPING RULE: Rewinding to epoch {} which had the highest validation accuracy: {}\n".format(best_epoch, best_accuracy))
                #self.__setstate__(best_state)

            self.tf_is_training = False #TODO: actually make sure this makes sense.


#TODO: PRIYA save functions providing the same function signature as previously
if __name__ == "__main__":
    pass
