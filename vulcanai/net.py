"""Contains the class for creating networks."""

import time
import os
import sys
import json
import numpy as np
import tensorflow as tf
from utils import get_timestamp
from ops import activations, optimizers

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
        #TODO: figure out how to deal with input_network.layer
        self.name = name
        self.layers = []
        #self.cost = None
        #self.val_cost = None #TODO: figure out if you really need these.
        self.input_dimensions = dimensions #TODO: I think for non-numeric this needs to contain more info
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

        self.network = self.create_network(
            config=self.config,
            nonlinearity=activations[self.activation]
        )
        if self.y is not None:
            self.trainer = self.create_trainer()
            self.validator = self.create_validator()

        #TODO: here we create our my_model function
        model_function = self.make_model_function() #TODO: fix param passing!

        #TODO: actually train this
        self.classifier = tf.estimator.Estimator(
            model_fn=model_function,
            params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })

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

        #alternatively make this a private top-level function...
        #could also call helper functions inside here...
        def my_model(features, labels, mode):

            network = self.network #TODO: better off as a function call? arghhh

            predicted_classes = tf.argmax(network, 1)

            #TODO: reimplement this
            # if mode == tf.estimator.ModeKeys.PREDICT:
            #     predictions = {
            #         'class_ids': predicted_classes[:, tf.newaxis],
            #         'probabilities': tf.nn.softmax(network),
            #         'logits': network,
            #     }
            #     return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            if self.num_classes is None or self.num_classes == 0:
                loss = tf.losses.mean_squared_error(labels=labels, logits=network)

            else:
                loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=network)

            # Compute evaluation metrics.
            # TODO: can configure this.
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
                updates = None
                ValueError("No optizer found") #TODO my goodness make sure you're still passing things

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    def create_network(self, config, nonlinearity):
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

        if self.num_classes is not None and self.num_classes != 0:
            network = self.create_classification_layer(
                network,
                num_classes=self.num_classes,
                nonlinearity=activations[self.pred_activation]
            )

        return network


    def create_dense_network(self, units, dropouts, nonlinearity):
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
        network = tf.feature_column.input_layer(self.features, self.params['feature_columns'])

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


    def cross_entropy_loss(self, prediction, y):
        """Generate a cross entropy loss function."""
        print("Using categorical cross entropy loss")
        return lasagne.objectives.categorical_crossentropy(prediction,
                                                           y).mean()

    def mse_loss(self, prediction, y):
        """Generate mean squared error loss function."""
        print("Using Mean Squared error loss")
        return lasagne.objectives.squared_error(prediction, y).mean()



    def create_validator(self):
        """
        Generate theano function to check error and accuracy of the network.

        Returns: theano function that takes input (train_x,train_y)
                 and returns error and accuracy
        """
        print("Creating {} Validator...".format(self.name))
        # create prediction
        val_prediction = lasagne.layers.get_output(
            self.network,
            deterministic=True
        )
        # check how much error in prediction
        if self.val_cost is None:
            if self.num_classes is None or self.num_classes == 0:
                self.val_cost = self.mse_loss(val_prediction, self.y)
                val_acc = T.constant(0)
            else:
                self.val_cost = self.cross_entropy_loss(val_prediction, self.y)
                # check the accuracy of the prediction
                if self.num_classes > 1:
                    val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1),
                                          T.argmax(self.y, axis=1)),
                                     dtype=theano.config.floatX)
                elif self.num_classes == 1:
                    val_acc = T.mean(T.eq(T.round(val_prediction,
                                                  mode='half_away_from_zero'),
                                          self.y),
                                     dtype=theano.config.floatX)

        return theano.function([self.input_var, self.y],
                               [self.val_cost, val_acc])


    def forward_pass(self, input_data, convert_to_class=False):
        """
        Allow the implementer to quickly get outputs from the network.

        Args:
            input_data: Numpy matrix to make the predictions on
            convert_to_class: If true, output the class
                             with highest probability

        Returns: Numpy matrix with the output probabilities
                 with each class unless otherwise specified.
        """
        if convert_to_class:
            return get_class(self.output(input_data))
        else:
            return self.output(input_data)


    #TODO: actually put this in __init__ of the model
    #taken from https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py
    def create_trainer(features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        #TODO: This is actually a batch ratio!
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        # Return the dataset.
        return dataset

    #taken from https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py
    def create_validator(features, labels, batch_size):
        """An input function for evaluation or prediction"""




                eval_result = self.classifier.evaluate(
        input_fn=lambda:self.eval_input_fn(test_x, test_y, args.batch_size))

        features=dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset


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


        try:
            for epoch in range(epochs):
                epoch_time = time.time()
                print("--> Epoch: {}/{}".format(
                        epoch,
                        epochs - 1
                ))

                #TODO: do we need to provide a value to the steps parameter?
                #https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/estimator/estimator.py
                #TODO: make this into a function!
                self.classifier.train(input_fn=lambda:self.train_input_fn(train_x, train_y, batch_ratio))

                train_error, train_accuracy = self.validator(train_x, train_y)
                validation_error, validation_accuracy = self.validator(val_x,val_y)

                if self.stopping_rule == 'best_validation_error' and validation_error < best_error:
                    best_state = self.__getstate__() #TODO: this is probably gonna break....
                    best_epoch = epoch
                    best_error = validation_error

                elif self.stopping_rule == 'best_validation_accuracy' and validation_accuracy > best_accuracy:
                    best_state = self.__getstate__() #TODO: this is probably gonna break
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
                self.__setstate__(best_state)

            elif self.stopping_rule == 'best_validation_accuracy':
                print("STOPPING RULE: Rewinding to epoch {} which had the highest validation accuracy: {}\n".format(best_epoch, best_accuracy))
                self.__setstate__(best_state)

            self.tf_is_training = False #TODO: actually make sure this makes sense.

    def __getstate__(self):
        """Pickle save config."""
        pickle_dict = dict()
        for k, v in self.__dict__.items():
            if not issubclass(v.__class__,
                              theano.compile.function_module.Function) \
                and not issubclass(v.__class__,
                                   theano.tensor.TensorVariable):
                pickle_dict[k] = v
        net_parameters = np.array(
            lasagne.layers.get_all_param_values(self.layers,
                                                **{self.name: True})
        )
        if self.input_network is None:
            return pickle_dict, net_parameters, None, None, None
        else:
            pickle_dict['input_network'] = None
            return (pickle_dict,
                    net_parameters,
                    self.input_network['network'].save_name,
                    self.input_network['layer'],
                    self.input_network['get_params'])

    def __setstate__(self, params):
        """Pickle load config."""
        self.__dict__.update(params[0])
        if params[2] is not None and params[3] is not None:
            input_network = Network.load_model(params[2])
            self.input_var = input_network.input_var
            self.input_network = {'network': input_network,
                                  'layer': params[3],
                                  'get_params': params[4]}
        else:
            tensor_size = len(self.input_dimensions)

            if tensor_size == 2:
                self.input_var = T.matrix('input')
            elif tensor_size == 3:
                self.input_var = T.tensor3('input')
            elif tensor_size == 4:
                self.input_var = T.tensor4('input')
            elif tensor_size == 5:
                self.input_var = T.tensor5('input')

        self.y = T.matrix('truth')
        self.__init__(self.__dict__['name'],
                      self.__dict__['input_dimensions'],
                      self.__dict__['input_var'],
                      self.__dict__['y'],
                      self.__dict__['config'],
                      self.__dict__['input_network'],
                      self.__dict__['num_classes'],
                      self.__dict__['activation'],
                      self.__dict__['pred_activation'],
                      self.__dict__['optimizer'],
                      self.__dict__['learning_rate'])
        lasagne.layers.set_all_param_values(self.layers,
                                            params[1],
                                            **{self.name: True})

    def save_model(self, save_path='models'):
        """
        Will save the model parameters to a npz file.

        Args:
            save_path: the location where you want to save the params
        """
        if self.input_network is not None:
            if not hasattr(self.input_network['network'], 'save_name'):
                self.input_network['network'].save_model()

        if not os.path.exists(save_path):
            print('Path not found, creating {}'.format(save_path))
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "{}{}".format(self.timestamp,
                                                          self.name))
        self.save_name = '{}.network'.format(file_path)
        print('Saving model as: {}'.format(self.save_name))

        with open(self.save_name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.save_metadata(file_path)

    @classmethod
    def load_model(cls, load_path):
        """
        Will load the model parameters from npz file.

        Args:
            load_path: the exact location where the model has been saved.
        """
        print('Loading model from: {}'.format(load_path))
        with open(load_path, 'rb') as f:
            instance = pickle.load(f)
        return instance

    def save_record(self, save_path='records'):
        """
        Will save the training records to file to be loaded up later.

        Args:
            save_path: the location where you want to save the records
        """
        if self.record is not None:
            if not os.path.exists(save_path):
                print('Path not found, creating {}'.format(save_path))
                os.makedirs(save_path)

            file_path = os.path.join(save_path, "{}{}".format(self.timestamp,
                                                              self.name))
            print('Saving records as: {}_stats.pickle'.format(file_path))
            with open('{}_stats.pickle'.format(file_path), 'w') as output:
                pickle.dump(self.record, output, -1)
        else:
            print("No record to save. Train the model first.")

    def save_metadata(self, file_path='models'):
        """
        Will save network configuration alongside weights.

        Args:
            file_path: the npz file path without the npz
        """
        config = {
            "{}".format(file_path): {
                "input_dimensions": self.input_dimensions,
                "input_var": "{}".format(self.input_var.type),
                "y": "{}".format(self.y.type),
                "config": self.config,
                "num_classes": self.num_classes,
                "input_network": {
                    'network': None,
                    'layer': None
                }
            }
        }

        if self.input_network:
            config["{}".format(file_path)]["input_network"]['network'] = \
                self.input_network['network'].save_name
            config["{}".format(file_path)]["input_network"]['layer'] = \
                self.input_network['layer']

        json_file = "{}_metadata.json".format(file_path)
        print('Saving metadata to {}'.format(json_file))
        with open(json_file, 'w') as f:
            json.dump(config, f)


if __name__ == "__main__":
    pass
