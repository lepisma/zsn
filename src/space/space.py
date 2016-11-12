"""
Word vector embedding space
"""

import numpy as np


class Space(object):
    """
    A wrapper for trained word vector model
    """

    def __init__(self, data_file):
        """
        Read output from C version of glove
        """

        self.words = []
        self.data = []

        f = open(data_file, "r")
        for line in f:
            items = line.split()
            self.words.append(items[0])
            self.data.append([float(item) for item in items[1:]])
        f.close()

        self.words = np.array(self.words)
        self.data = np.array(self.data)

        # Precomputing vector norms
        self.norm = np.linalg.norm(self.data, axis=1)

    def get_nearest_words(self, vector, count):
        """
        Return the nearest 'count' words with confidence for the given vector
        """

        similarity = np.divide(np.dot(self.data, vector),
                               (self.norm * np.linalg.norm(vector)))

        indices = np.argsort(similarity)

        return self.words[indices[-count:]][::-1]

    def get_vector(self, word):
        """
        Return vector of given word
        """

        return self.data[np.where(self.words == word)[0][0]]
