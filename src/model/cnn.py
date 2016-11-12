"""
Models
"""

import numpy as np
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD


class Projection(object):
    """
    Learns projections from dense hidden layer of CNN to word space.
    Sizes assume the usual cases for CIFAR
    """

    def __init__(self, Xim_shape, V_shape):

        model = Sequential()
        model.add(Dense(100, input_dim=Xim_shape, activation="tanh"))
        model.add(Dropout(0.5))
        model.add(Dense(V_shape))

        self.model = model

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss="mse", optimizer=sgd)

    def train(self, Xim_train, V_train, batch_size, nb_epoch):
        """
        Train the projection
        """

        self.model.fit(Xim_train, V_train, batch_size=batch_size, nb_epoch=nb_epoch)

    def predict(self, X):
        """
        Return the projection vector
        """

        return self.model.predict(X)

class CNNProj(object):
    """
    Parent CNN + Projection class
    """

    def __init__(self, name, classes, X_shape, Xim_shape, Y_shape, V_shape):

        self.name = name
        self.classes = classes

        # VGG Style model in graph
        model = Sequential()
        model.add(Convolution2D(32, 3, 3,
                                activation="relu",
                                border_mode="full",
                                input_shape=X_shape))
        model.add(Convolution2D(32, 3, 3, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, 3, 3,
                                activation="relu",
                                border_mode="full"))
        model.add(Convolution2D(64, 3, 3, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(Xim_shape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(Y_shape, activation="relu"))
        model.add(Activation("softmax"))

        self.model = model

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss="categorical_crossentropy", optimizer=sgd)

        # Projection
        self.proj = Projection(Xim_shape, V_shape)

        # Function to get Xim
        self.get_Xim = theano.function([self.model.layers[0].input],
                                       model.layers[10].get_output(train=False))

    def train_softmax(self, X_train, Y_train, batch_size, nb_epoch):
        """
        Train the main classification model
        """

        self.model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)

    def train_embedding(self, X_train, V_train, batch_size, nb_epoch):
        """
        Train projection
        - Generate Xim_train using trained main model
        """

        Xim_train = self.get_Xim(X_train)
        self.proj.train(Xim_train, V_train, batch_size, nb_epoch)

    def train(self, X_train, Y_train, V_train, batch_size_list, nb_epoch_list):
        """
        Train the main model. Then tune the projection
        Batch size and number of epoch lists have elements corresponding to
        softmax and embedding repectively
        """

        self.train_softmax(X_train, Y_train, batch_size_list[0], nb_epoch_list[0])
        self.train_embedding(X_train, V_train, batch_size_list[1], nb_epoch_list[1])

    def predict(self, X):
        """
        Return the softmax and embedding output
        """

        Y = self.model.predict(X)
        V = self.proj.predict(self.get_Xim(X))

        return [Y, V]

    def accuracies(self, X_train, Y_train, V_train, X_test, Y_test, V_test):
        """
        Return  the current accuracy
        """

        accuracies = {}

        accuracies["Training accuracy"] = self.model.evaluate(X_train, Y_train, show_accuracy=True)[1]
        accuracies["Testing accuracy"] = self.model.evaluate(X_test, Y_test, show_accuracy=True)[1]

        accuracies["Training accuracy (embedding)"] = self.proj.model.evaluate(self.get_Xim(X_train), V_train, show_accuracy=True)[1]
        accuracies["Testing accuracy (embedding)"] = self.proj.model.evaluate(self.get_Xim(X_test), V_test, show_accuracy=True)[1]

        return accuracies
