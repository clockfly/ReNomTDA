# -*- coding: utf-8 -*-
import numpy as np

from sklearn import decomposition, manifold


class L1Centrality(object):
    """Class of L1 Centrality lens.
    """

    def __init__(self):
        pass

    def fit_transform(self, data):
        """Function of projection data to L1 centrality axis.

        Params:
            data: distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        return np.sum(data, axis=1).reshape(data.shape[0], 1)


class LinfCentrality(object):
    """Class of projection data to L-infinity centrality axis.
    """

    def __init__(self):
        pass

    def fit_transform(self, data):
        """Function of projection data to L-inf Centrality axis.

        Params:
            data: distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        return np.max(data, axis=1).reshape(data.shape[0], 1)


class GaussianDensity(object):
    """Class of projection data to gaussian density axis.

    Params:
        h: The width of kernel.
    """

    def __init__(self, h=0.3):
        if h == 0:
            raise Exception("Parameter h must not zero.")

        self.h = h

    def fit_transform(self, data):
        """Function of projection data to Gaussian Density axis.

        Params:
            data: distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        return np.sum(np.exp(-(data**2 / (2 * self.h))), axis=1).reshape(data.shape[0], 1)


class PCA(object):
    """Class of projection data to PCA components axis.

    Params:
        components: The axis of projection. If you use compoents 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1]):
        if components is None:
            raise Exception("Component error.")

        self.components = components
        self.pca = decomposition.PCA(n_components=(max(self.components) + 1))

    def fit_transform(self, data):
        """Function of projection data to PCA axis.

        Params:
            data: raw data or distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        projected_data = self.pca.fit_transform(data)
        self.contribution_ratio = self.pca.explained_variance_ratio_[self.components]
        self.axis = self.pca.components_
        return projected_data[:, self.components]


class TSNE(object):
    """Class of projection data to TSNE components axis.

    Params:
        components: The axis of projection. If you use compoents 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1]):
        if components is None:
            raise Exception("Component error.")

        self.components = components
        self.tsne = manifold.TSNE(n_components=max(components) + 1, init='pca')

    def fit_transform(self, data):
        """Function of projection data to TSNE axis.

        Params:
            data: raw data or distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        projected_data = self.tsne.fit_transform(data)
        return projected_data[:, self.components]


class MDS(object):
    """Class of projection data to MDS components axis.

    Params:
        components: The axis of projection. If you use compoents 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1]):
        if components is None:
            raise Exception("Component error.")

        self.components = components
        self.mds = manifold.MDS(n_components=max(components) + 1, random_state=10)

    def fit_transform(self, data):
        """Function of projection data to MDS axis.

        Params:
            data: raw data or distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        projected_data = self.mds.fit_transform(data)
        return projected_data[:, self.components]
