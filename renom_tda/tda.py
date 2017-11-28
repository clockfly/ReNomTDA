# -*- coding: utf-8 -*.t-
from __future__ import print_function

# import colorsys

# import itertools

# import matplotlib.pyplot as plt

# import networkx as nx

import numpy as np

# import scipy

from sklearn import cluster, decomposition, preprocessing


class TDA(object):
    """
    TDA class
    """

    def __init__(self, verbose=1):
        """
        init
        """
        self.verbose = verbose
        self.src_data = {
            "text_data": None,
            "numerical_data": None,
            "target_data": None
        }
        self.params = {
            "metric": None,
            "lens": None,
            "scaler": None,
            "resolution": 0,
            "overlap": 0
        }
        self.point_cloud_data = {
            "nodes": None,
            "colors": None,
            "sizes": None
        }
        self.topology_data = {
            "hypercubes": None,
            "nodes": None,
            "edges": None,
            "colors": None,
            "sizes": None
        }
        pass

    def load_data(self, text_data=None, numerical_data=None):
        """
        load data
        """
        if type(numerical_data) is not np.ndarray:
            numerical_data = np.array(numerical_data)

        if numerical_data.ndim != 2:
            raise ValueError("Data must be 2d array.")

        self.src_data["text_data"] = text_data
        self.src_data["numerical_data"] = numerical_data

    def fit_transform(self, metric=None, lens=None, scaler=preprocessing.MinMaxScaler()):
        """
        create point cloud
        """
        if self.src_data["numerical_data"] is None:
            raise Exception("Data must not None.")

        self.params["metric"] = metric
        self.params["lens"] = lens
        self.params["scaler"] = scaler

        data = self.src_data["numerical_data"]

        # metricをもとに距離行列を作る。
        if (metric is not None) and ("fit_transform" in dir(metric)):
            if self.verbose == 1:
                print("calculated distance matrix by %s class using %s distance." %
                      (self.params["metric"].__class__.__name__, self.params["metric"].metric))
            dist_matrix = metric.fit_transform(data)
        else:
            # metricがNoneならdataをそのまま使う
            dist_matrix = data

        # lensによって射影後の次元数は異なるのでNoneで初期化
        projected_data = None
        if (lens is not None) and (len(lens) > 0):
            for l in lens:
                if (l is not None) and "fit_transform" in dir(l):
                    if self.verbose == 1:
                        print("projected by %s." % (l.__class__.__name__))

                    if projected_data is None:
                        projected_data = l.fit_transform(dist_matrix)
                    else:
                        p = l.fit_transform(dist_matrix)
                        projected_data = np.concatenate([projected_data, p], axis=1)
                else:
                    projected_data = dist_matrix
        else:
            # lensがNoneならdist_matrixをそのまま使う
            projected_data = dist_matrix

        # scalerで正規化。
        if projected_data is not None:
            if (scaler is not None) and ("fit_transform" in dir(scaler)):
                projected_data = scaler.fit_transform(projected_data)

        self.point_cloud_data["nodes"] = projected_data

        if self.verbose == 1:
            print("finish fit_transform.")
        pass

    def map(self, resolution=10, overlap=0.5, clusterer=cluster.DBSCAN(eps=1, min_samples=1)):
        """
        mapping point cloud to topological space
        """
        pass

    def color(self, target, dtype="numerical", ctype="rgb", feature_range=(0, 1), normalized=False):
        """
        color topology
        """
        pass

    # 元データに対して前処理
    def get_statistic_values(self):
        """
        statistic values
        """
        pass

    def get_histogram_data(self):
        """
        histogram
        """
        pass

    def select_data(self):
        """
        data selection.
        """
        pass

    def preprocess_data(self):
        """
        preprocessing
        """
        pass

    # point cloudに関する処理
    def color_point_cloud(self):
        pass

    def calc_size_point_cloud(self):
        pass

    def clustering_point_cloud(self, supervised=False, method=None):
        pass

    def show_point_cloud(self):
        pass

    def save_point_cloud(self):
        pass

    def search(self):
        pass


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from renom_tda.lens import PCA

    iris = load_iris()
    data = iris.data
    target = iris.target

    tda = TDA()
    tda.load_data(numerical_data=data)
    tda.fit_transform(metric=None, lens=[PCA(components=[0, 1])])
    print(tda.point_cloud_data["nodes"].shape)
