"""Contains the class for creating networks."""

import time
import os
import sys
import json
import numpy as np
import tensorflow as tf
from utils import get_timestamp
from ops import activations, optimizers
from sklearn.utils import shuffle


sys.setrecursionlimit(5000)

class Network(object):
    """Class to generate networks and train them."""

    def __init__(self, name, dimensions, input_var, y, config,
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
        #TODO: really this should be named
        #TODO: change the dimension hackkkkkk
        self.feature_columns = [tf.feature_column.numeric_column(key=x) for x in range(self.input_dimensions[1])]

        #TODO: this whole structure and param passing are dumb
        model_function = self.make_model_function()

        self.classifier = tf.estimator.Estimator(model_fn=model_function)

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
        self.minibatch_iteration = 0 #TODO: may need to change depending on what functions are called


    def make_model_function(self):

        #alternatively make this a "private" top-level function...
        #could also call helper functions inside here...
        def my_model(features, labels, mode):

            network = self.create_network(features, config=self.config, nonlinearity=activations[self.activation]) #TODO: better off as a function call? arghhh

            print network
            predicted_classes = tf.argmax(network, 1)

            #TODO: reimplement this
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_ids': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(network),
                    'logits': network,
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            if self.num_classes is None or self.num_classes == 0:
                loss = tf.losses.mean_squared_error(labels=labels, logits=network)

            else:
                loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=network) #TODO: maybe not one_hot??

            # Compute evaluation metrics.
            # TODO: can configure this.
            print predicted_classes
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=predicted_classes,
                                           name='acc_op')
            metrics = {'accuracy': accuracy}
            tf.summary.scalar('accuracy', accuracy[1])

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            # Create training op.
            #this was taken from the internetz. I assume it's so an exception is thrown instead of return
            #don't really like it, but can change later.
            assert mode == tf.estimator.ModeKeys.TRAIN

            #TODO: need to pass the learning rate var appropriately!
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            elif self.optimizer == 'sgd':
                optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            else:
                ValueError("No optizer found") #TODO my goodness make sure you're still passing things

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step()) #TODO: work with global step for stopped.
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

        # elif mode == 'conv':
        #     jsonschema.validate(config, schemas.conv_network)
        #     network = self.create_conv_network(
        #         filters=config.get('filters'),
        #         filter_size=config.get('filter_size'),
        #         stride=config.get('stride'),
        #         pool_mode=config['pool'].get('mode'),
        #         pool_stride=config['pool'].get('stride'),
        #         nonlinearity=nonlinearity
        #     )
        else:
            raise ValueError('Mode {} not supported.'.format(mode))

        if self.num_classes is not None and self.num_classes != 0:
            network = self.create_classification_layer(
                network,
                num_classes=self.num_classes,
                nonlinearity=activations[self.pred_activation]
            )

        return network


    def create_dense_network(self, features, units, dropouts, nonlinearity):
        """
        Generate a dense network.

        Args:
            units: The list of number of nodes to have at each layer
            dropouts: The list of dropout probabilities for each layer
            nonlinearity: Nonlinearity from Lasagne.nonlinearities

        Returns: the output of the network (linked up to all the layers)
        """

        if len(units) != len(dropouts):
            raise ValueError(
                "Cannot build network: units and dropouts don't correspond"
            )

        #TODO: this should be logger
        print("Creating {} Network...".format(self.name))

        #TODO: here integrate the whole input network thing

        #TODO: where features is a mapping from key to tensor and features columns is an iterable
        #TODO: actually get this from init params


        network = tf.layers.dense(features, self.input_dimensions) #TODO: this is almost definitely wrong

        #TODO: you probably have to do something different for selu?? check.
        #TODO: name layers?
        for i, (num_units, prob_dropout) in enumerate(zip(units, dropouts)):
            network = tf.layers.dense(network, units=num_units, activation=nonlinearity)
            network = tf.layers.dropout(network, rate=prob_dropout, training=self.tf_is_training)

        return network


    def create_classification_layer(self, network, num_classes, nonlinearity):
        """
        Create a classification layer. Normally used as the last layer.

        Args:
            network: network you want to append a classification to
            num_classes: how many classes you want to predict
            nonlinearity: nonlinearity to use as a string (see DenseLayer)

        Returns: the classification layer appended to all previous layers
        """
        print('\tOutput Layer:')

        network = tf.layers.dense(network, num_classes, activation=nonlinearity)

        #TODO: fix this to be whatever it is in tensorflow
        #print('\t\t{}'.format(lasagne.layers.get_output_shape(network)))

        return network


    def eval_input_fn(self, features, labels, batch_ratio):
        """An input function for evaluation or prediction"""
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        #TODO: fix this hack self.input_dimensions[1]
        batch_size = int(self.input_dimensions[1] * batch_ratio) #TODO: better check this is right

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset


    #TODO: think about declaring and storing this in advance
    def train_input_fn(self, features, labels, batch_ratio):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)) #TODO: this will probably break

        #TODO: fix this hack self.input_dimensions[1]
        batch_size = int(self.input_dimensions[1] * batch_ratio) #TODO: better check this is right

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        # Return the read end of the pipeline.
        return dataset.make_one_shot_iterator().get_next()

    def train(self, epochs, train_x, train_y, val_x, val_y,
              batch_ratio=0.1, plot=True, change_rate=None):
        """
        Train the network.

        Args:
            epochs: how many times to iterate over the training data
            train_x: the training data
            train_y: the training truth
            val_x: the validation data (should not be also in train_x)
            val_y: the validation truth (should not be also in train_y)
            batch_ratio: the percent (0-1) of how much data a batch should have
            plot: If True, plot performance during training
            change_rate: a function that updates learning rate (takes an alpha, returns an alpha)'

        """

        print('\nTraining {} in progress...\n'.format(self.name))
        self.tf_is_training = True #for dropout

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
            best_state = None
            best_epoch = None
            best_error = float('inf')

        elif self.stopping_rule == 'best_validation_accuracy':
            best_state = None
            best_epoch = None
            best_accuracy = 0.0

        #TODO: catch all the errors!!!
        #TODO: need to be able to update the learning rate. I think you need to make it a tensor object in your model fun optimizer
        #https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer
        #https://github.com/tensorflow/tensorflow/issues/2198

        try:
            for epoch in range(epochs):
                epoch_time = time.time()
                print("--> Epoch: {}/{}".format(
                        epoch,
                        epochs - 1
                ))

                #TODO: I think I still need to leave this in?
                train_x, train_y = shuffle(train_x, train_y, random_state=0)

                self.classifier.train(input_fn=lambda:self.train_input_fn(train_x, train_y, batch_ratio))


                #TODO: somehow get percentage batch finished


                #TODO: do we need to provide a value to the steps parameter?
                #https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/estimator/estimator.py
                #TODO: make this into a function!

                train_eval_result = self.classifier.evaluate(input_fn=lambda:self.eval_input_fn(train_x, train_y, batch_ratio))
                val_eval_result = self.classifier.evaluate(input_fn=lambda:self.eval_input_fn(val_x, val_y, batch_ratio))

                train_error = train_eval_result["loss"]
                train_accuracy = train_eval_result["accuracy"]
                validation_error = val_eval_result["loss"]
                validation_accuracy = val_eval_result['accuracy']

                if self.stopping_rule == 'best_validation_error' and validation_error < best_error:
                    #best_state = self.__getstate__() #TODO: this is probably gonna break....
                    best_epoch = epoch
                    best_error = validation_error

                elif self.stopping_rule == 'best_validation_accuracy' and validation_accuracy > best_accuracy:
                    #best_state = self.__getstate__() #TODO: this is probably gonna break
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

if __name__ == "__main__":
    pass
