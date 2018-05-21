# -*- coding: utf-8 -*-
from __future__ import print_function
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass
import functools
import numpy as np
from sklearn import decomposition, manifold


class Lens(with_metaclass(ABCMeta, object)):
    """Abstract class of data loading modules."""

    def _check_data(func):
        """Function of check input data."""
        @functools.wraps(func)
        def wrapper(*args):
            if args[1] is None:
                raise ValueError("Input data is None.")
            return func(args[0], np.array(args[1]))
        return wrapper

    @abstractmethod
    def fit_transform(self):
        """fit_transform is required."""
        pass


class L1Centrality(Lens):
    """Class of L1 Centrality lens."""

    def __init__(self):
        pass

    @Lens._check_data
    def fit_transform(self, data):
        """Function of projection data to L1 centrality axis.

        Params:
            data: distance matrix.
        """
        return np.sum(data, axis=1).reshape(data.shape[0], 1)


class LinfCentrality(Lens):
    """Class of projection data to L-infinity centrality axis."""

    def __init__(self):
        pass

    @Lens._check_data
    def fit_transform(self, data):
        """Function of projection data to L-inf Centrality axis.

        Params:
            data: distance matrix.
        """
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

    @Lens._check_data
    def fit_transform(self, data):
        """Function of projection data to Gaussian Density axis.

        Params:
            data: distance matrix.
        """
        return np.sum(np.exp(-(data**2 / (2 * self.h))), axis=1).reshape(data.shape[0], 1)


class PCA(Lens):
    """Class of projection data to PCA components axis.

    Params:
        components: The axis of projection. If you use compoents 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1]):
        if components is None:
            raise Exception("Component error.")

        self.components = components
        self.pca = decomposition.PCA(n_components=(max(self.components) + 1))

    @Lens._check_data
    def fit_transform(self, data):
        """Function of projection data to PCA axis.

        Params:
            data: raw data or distance matrix.
        """
        projected_data = self.pca.fit_transform(data)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_[self.components]
        self.components_ = self.pca.components_[:, self.components]
        return projected_data[:, self.components]


class TSNE(Lens):
    """Class of projection data to TSNE components axis.

    Params:
        components: The axis of projection. If you use compoents 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1]):
        if components is None:
            raise Exception("Component error.")

        self.components = components
        self.tsne = manifold.TSNE(n_components=max(components) + 1, init='pca')

    @Lens._check_data
    def fit_transform(self, data):
        """Function of projection data to TSNE axis.

        Params:
            data: raw data or distance matrix.
        """
        projected_data = self.tsne.fit_transform(data)
        return projected_data[:, self.components]


class MDS(Lens):
    """Class of projection data to MDS components axis.

    Params:
        components: The axis of projection. If you use compoents 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1]):
        if components is None:
            raise Exception("Component error.")

        self.components = components
        self.mds = manifold.MDS(n_components=max(components) + 1, random_state=10)

    @Lens._check_data
    def fit_transform(self, data):
        """Function of projection data to MDS axis.

        Params:
            data: raw data or distance matrix.
        """
        projected_data = self.mds.fit_transform(data)
        return projected_data[:, self.components]


class Isomap(Lens):
    """Class of projection data to MDS components axis.

    Params:
        components: The axis of projection. If you use compoents 0 and 1, this is [0, 1].
    """
    def __init__(self, components=[0, 1]):
        if components is None:
            raise Exception("Component error.")

        self.components = components
        self.isomap = manifold.Isomap(n_components=max(components) + 1)

    @Lens._check_data
    def fit_transform(self, data):
        """Function of projection data to MDS axis.

        Params:
            data: raw data or distance matrix.
        """
        projected_data = self.isomap.fit_transform(data)
        return projected_data[:, self.components]


class Lenses(Lens):
    """Class of projection data.

    Params:
        filters: The array of lens.
    """

    def __init__(self, filters):
        self.filters = filters

    @Lens._check_data
    def fit_transform(self, data):
        """Function of projection with filters.

        Params:
            data: raw data or distance matrix.

        """
        # initialize return data.
        ret_data = None

        if (self.filters is not None) and (len(self.filters) > 0):
            for f in self.filters:
                if (f is not None) and "fit_transform" in dir(f):
                    if ret_data is None:
                        ret_data = f.fit_transform(data)
                    else:
                        p = f.fit_transform(data)
                        ret_data = np.concatenate([ret_data, p], axis=1)
            return ret_data
        else:
            return data
