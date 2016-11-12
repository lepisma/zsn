"""
Generate data for CIFAR100
"""

import numpy as np
from keras.datasets import cifar100
from keras.utils import np_utils
import cPickle

import sys
sys.path.insert(0, "../../src")

from space.space import Space

vector_dim = 50

# Load embedding space
emb = Space("../../data/glove.6B." + str(vector_dim) + "d.txt")

# Load data
(X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

# Load labels
labels = cPickle.load(open("../../data/meta", "rb"))["fine_label_names"]

# Get label vectors
V_train = np.zeros((X_train.shape[0], vector_dim))
V_test = np.zeros((X_test.shape[0], vector_dim))

remove_classes = []

for idx, row in enumerate(Y_train):
    if row[0] not in remove_classes:
        try:
            V_train[idx, :] = emb.get_vector(labels[row[0]])
        except IndexError:
            # Vector not found, mark class for removal
            remove_classes.append(row[0])

for idx, row in enumerate(Y_test):
    if row[0] not in remove_classes:
        V_test[idx, :] = emb.get_vector(labels[row[0]])

# Remove classes with no vectors
train_indices = ~np.in1d(Y_train, remove_classes)
test_indices = ~np.in1d(Y_test, remove_classes)

X_train = X_train[train_indices, :]
Y_train = Y_train[train_indices, :]
V_train = V_train[train_indices, :]

X_test = X_test[test_indices, :]
Y_test = Y_test[test_indices, :]
V_test = V_test[test_indices, :]

# Dump data
data = {
    "labels": labels,
    "data": [(X_train, Y_train, V_train), (X_test, Y_test, V_test)]
}

cPickle.dump(data, open("../../data/cifar100_with_vec.pkl", "w"))
