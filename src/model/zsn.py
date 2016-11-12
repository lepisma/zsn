"""
Main inference model
"""


import numpy as np
import pyprind


class ZSN(object):

    def __init__(self, cnns, space):
        """
        Use the given CNNs and embedding space
        """
        self.cnns = cnns
        self.space = space

    def predict(self, X):
        """
        Return embedding outputs for given input
        """

        network_outputs = []
        for cnn in self.cnns:
            network_outputs.append(cnn.predict(X)[1])
        average_embedding = np.mean(np.array(network_outputs), axis=0)
        return [average_embedding, np.array(network_outputs)]

    def evaluate_zero_shot_sim(self, X_unknown, V_unknown):
        """
        Evaluate zero shot similarity performance using cosine distance
        """

        [average_embedding, network_outputs] = self.predict(X_unknown)

        # global_dist = []
        # indi_dist = [[] for i in xrange(len(self.cnns))]
        # v_norm = np.linalg.norm(V_unknown, axis=1)

        # bar = pyprind.ProgBar(len(X_unknown))
        # for idx, x in enumerate(X_unknown):
        #     res = self.predict(x)
        #     global_dist.append(np.divide(np.dot(res["vec"], V_unknown[idx]),
        #                                 (np.linalg.norm(res["vec"]) * v_norm[idx])))

        #     tmp = res["indi_vec"]
        #     for i in xrange(len(self.cnns)):
        #         indi_dist[i].append(np.divide(np.dot(tmp[i], V_unknown[idx]),
        #                                       (np.linalg.norm(tmp[i]) * v_norm[idx])))

        #     bar.update()

        # return [global_dist, indi_dist]
