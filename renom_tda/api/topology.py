# -*- coding: utf-8 -*.t-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster, preprocessing

from renom_tda import Lenses
from renom_tda import DistUtil, MapUtil, GraphUtil
from renom_tda import PainterResolver
from renom_tda import PresenterResolver
from renom_tda import SearcherResolver


class Topology(object):
    """Class of Topology.

    Params:
        verbose: Show progress or not. 0(don't show), 1(show).
    """
    def __init__(self, verbose=1):
        self.verbose = verbose
        self._init_params()

    def _init_params(self):
        # input data
        self.text_data = None
        self.text_data_columns = None
        self.number_data = None
        self.number_data_columns = None
        # standardize info
        self.standardize = False
        self.std_number_data = None
        # transform
        self.dist_util = None
        self.metric = None
        self.lens = None
        self.scaler = None
        # point cloud
        self.point_cloud = None
        self.point_cloud_colors = None
        self.point_cloud_hex_colors = None
        # clustering train index
        self.train_index = np.array([])
        # map
        self.resolution = 1
        self.overlap = 1
        self.eps = 1
        self.min_samples = 1
        # hypercubes
        self.hypercubes = {}
        # graph_util
        self.graph_util = None
        self.nodes = None
        self.node_sizes = None
        self.edges = None
        self.colors = None
        self.hex_colors = None

    def load_data(self, number_data, text_data=None, number_data_columns=None,
                  text_data_columns=None, standardize=False):
        """Function of setting data to this instance.

        Params:
            number_data: Data using calclate topology.

            text_data: Text data correspond to number data.

            number_data_columns: Column names of number data.

            text_data_columns: Column names of text data.

            standardize: standardize number data or not.
        """
        # required number_data
        if number_data is None:
            raise ValueError("Number data must not None.")

        # number_data to numpy array
        if type(number_data) is not np.ndarray:
            number_data = np.array(number_data)

        # raise error when number_data dim != 2
        if number_data.ndim != 2:
            raise ValueError("Number data must be 2d array.")

        # text_data to numpy array
        if text_data is not None and type(text_data) is not np.ndarray:
            text_data = np.array(text_data)

        # text_data required, when text_data_columns exists
        if text_data_columns is not None and text_data is None:
            raise ValueError("When you input text data columns, text data must be exist.")

        # text_data_columns to numpy array
        if text_data_columns is not None and type(text_data_columns) is not np.ndarray:
            text_data_columns = np.array(text_data_columns)

        # text_data_columns and text_data must has same size
        if text_data_columns is not None and text_data is not None:
            if text_data.shape[1] != text_data_columns.shape[0]:
                raise ValueError("Text data and text data columns must have same number of columns.")

        # number_data_columns to numpy array
        if number_data_columns is not None and type(number_data_columns) is not np.ndarray:
            number_data_columns = np.array(number_data_columns)

        self.standardize = standardize

        self.number_data = number_data
        self.number_data_columns = number_data_columns
        self.text_data = text_data
        self.text_data_columns = text_data_columns

        if standardize:
            scaler = preprocessing.StandardScaler()
            self.std_number_data = scaler.fit_transform(self.number_data)
        else:
            self.std_number_data = number_data

    def load(self, loader, standardize=True):
        """Function of setting data to this instance with loader.

        Params:
            loader: Instance of Loader class.

            standardize: standardize number data or not.
        """
        self.standardize = standardize

        # loading data with load function of loader
        self.number_data, self.text_data, self.number_data_columns, self.text_data_columns = loader.load()

        if standardize:
            scaler = preprocessing.StandardScaler()
            self.std_number_data = scaler.fit_transform(self.number_data)
        else:
            self.std_number_data = np.array(self.number_data)

    def fit_transform(self, metric=None, lens=None,
                      scaler=preprocessing.MinMaxScaler()):
        """Function of projecting data to point cloud.

        Params:
            metric: distance metric. None, "euclidean", "cosine", etc.

            lens: List of projection filters.

            scaler: Class of normalize scaler.
        """
        if self.std_number_data is None:
            raise Exception("Data doesn't loaded.")

        self.metric = metric
        self.lens = lens
        self.scaler = scaler

        # calc dist matrix
        if metric is None:
            d = self.std_number_data
        else:
            self.dist_util = DistUtil(metric=metric)
            d = self.dist_util.calc_dist_matrix(self.std_number_data)

        # transform data
        if lens is not None:
            self.lenses = Lenses(filters=lens)
            d = self.lenses.fit_transform(d)

        # scaling data
        if (scaler is not None) and ("fit_transform" in dir(scaler)):
            d = scaler.fit_transform(d)

        self.point_cloud = d

    def map(self, resolution=10, overlap=1, eps=1, min_samples=1):
        """Function of mapping point cloud to topological space.

        Params:
            resolution: The number of division of each axis.

            overlap: The width of overlapping of division.

            eps: The distance of clustering of data in each division.

            min_samples: The least number of each cluster.
        """
        if self.point_cloud is None:
            raise Exception("Point cloud is None.")

        if resolution <= 0:
            raise Exception("Resolution must greater than 0.")

        if overlap <= 0:
            raise Exception("Overlap must greater than 0.")

        if eps <= 0:
            raise Exception("Eps must greater than 0.")

        if min_samples <= 0:
            raise Exception("Min samples must greater than 0.")

        self.resolution = resolution
        self.overlap = overlap
        self.eps = eps
        self.min_samples = min_samples

        # If dist util doesn't exists, using euclidean distance.
        if self.dist_util is None:
            self.dist_util = DistUtil()
            self.dist_util.calc_dist_matrix(self.std_number_data)
        eps_ = self.dist_util.calc_eps(eps)

        # create hypercubes with mapping utility
        clusterer = cluster.DBSCAN(eps=eps_, min_samples=min_samples)
        self.mapper = MapUtil(resolution=resolution, overlap=overlap, clusterer=clusterer)
        self.hypercubes = self.mapper.map(self.std_number_data, self.point_cloud)

        # create nodes and edges with graph utility
        self.graph_util = GraphUtil(point_cloud=self.point_cloud, hypercubes=self.hypercubes)
        self.nodes, self.node_sizes = self.graph_util.calc_node_coordinate()
        if self.verbose == 1:
            print("created {} nodes.".format(self.nodes.shape[0]))

        if len(self.nodes) < 2:
            raise Exception("Can't create node, please change parameters.")

        self.edges = self.graph_util.calc_edges()
        if self.verbose == 1:
            print("created {} edges.".format(len(self.edges)))

    def color(self, target, color_method="mean", color_type="rgb",
              normalize=True):
        """Function of coloring topology.

        Params:
            target: Array of coloring value.

            color_method: Method of calculate color value. "mean" or "mode".

            color_type: "rgb" or "gray".

            normalize: Normalize target data or not.
        """
        if self.point_cloud.shape[0] != len(target):
            raise Exception("target must have same row size of data.")

        if color_method not in ["mean", "mode"]:
            raise Exception("color_method {} is not usable.".format(color_method))

        if color_type not in ["rgb", "gray"]:
            raise Exception("color_type {} is not usable.".format(color_type))

        if normalize:
            scaler = preprocessing.MinMaxScaler()
            self.normalized_target = scaler.fit_transform(np.array(target).reshape(-1, 1).astype(float))
        else:
            self.normalized_target = np.array(target)

        self.colors, self.hex_colors = self.graph_util.color(self.normalized_target, color_method, color_type)

    def _get_presenter(self, fig_size, node_size, edge_width, mode, strength):
        """Function of resolve presenter."""
        resolver = PresenterResolver(fig_size, node_size, edge_width, strength)
        return resolver.resolve(mode)

    def show(self, fig_size=(5, 5), node_size=5, edge_width=1, mode="normal", strength=None):
        """Function of show topology.

        Params:
            fig_size: The size of showing figure.

            node_size: The size of node.

            edge_width: The width of edge.

            mode: Layout mode. "normal" or "spring".

            strength: Strength of repulsive force between nodes in "spring" mode.
        """
        if mode not in ["normal", "spring", "spectral"]:
            raise Exception("mode {} is not usable.".format(mode))

        # show with presenter show function.
        presenter = self._get_presenter(fig_size, node_size, edge_width, mode, strength)
        presenter.show(self.nodes, self.edges, self.node_sizes, self.hex_colors)

    def save(self, file_name, fig_size=(5, 5), node_size=5, edge_width=1, mode="normal", strength=None):
        """Function of show topology.

        Params:
            file_name: The name of output file.

            fig_size: The size of showing figure.

            node_size: The size of node.

            edge_width: The width of edge.

            mode: Layout mode. "normal" or "spring".

            strength: Strength of repulsive force between nodes in "spring" mode.
        """
        if mode not in ["normal", "spring"]:
            raise Exception("mode {} is not usable.".format(mode))

        # save with presenter save function.
        presenter = self._get_presenter(fig_size, node_size, edge_width, mode, strength)
        presenter.save(file_name, self.nodes, self.edges, self.node_sizes, self.hex_colors)

    def _node_index_from_data_id(self, data_index):
        # return node index that has data in args.
        node_index = []
        values = self.hypercubes.values()
        for i, val in enumerate(values):
            s1 = set(val)
            s2 = set(data_index)
            if len(s1.intersection(s2)) > 0:
                node_index.append(i)
        return node_index

    def _set_search_color(self, node_index):
        # set searched color.
        searched_color = ["#cccccc"] * len(self.hypercubes.keys())
        for i in node_index:
            searched_color[i] = self.hex_colors[i]
        self.hex_colors = searched_color

    def search_from_values(self, search_dicts=None, target=None,
                           search_type="and"):
        """Function of search topology. This function is old version.

        Params:
            search_dicts: dictionaly of search conditions.

            target: append search data.

            search_type: "and" or "or".
        """
        if search_type not in ["and", "or"]:
            raise Exception("search_type {} is not usable.".format(search_type))

        self.search(search_dicts=search_dicts, target=target, search_type=search_type)

    def _get_searched_index(self, data=None, search_dicts=None, search_type="and"):
        """Function of getting data index with search conditions."""
        resolver = SearcherResolver(data, self.text_data, self.number_data_columns, self.text_data_columns)

        data_index = []

        for d in search_dicts:
            # resolve searcher from data_type.
            searcher = resolver.resolve(d["data_type"])

            # get data index with searcher search function.
            index = searcher.search(d["column"], d["operator"], d["value"])

            # concat data_index and index.
            if len(data_index) > 0:
                if len(index) > 0:
                    s1 = set(data_index)
                    s2 = set(index)

                    if search_type == "and":
                        data_index = list(s1.intersection(s2))
                    elif search_type == "or":
                        data_index = list(s1.union(s2))
            else:
                data_index = index
        return data_index

    def search(self, search_dicts=None, target=None, search_type="and"):
        """Function of search topology.

        Params:
            search_dicts: dictionaly of search conditions.

            target: append search data.

            search_type: "and" or "or".
        """
        if search_type not in ["and", "or"]:
            raise Exception("search_type {} is not usable.".format(search_type))

        if target is None:
            d = self.number_data
        else:
            if self.number_data.shape[0] != len(target):
                raise Exception("target must have same row size of data.")
            d = np.concatenate([self.number_data, target.reshape(-1, 1)], axis=1)

        data_index = self._get_searched_index(data=d, search_dicts=search_dicts, search_type=search_type)
        node_index = self._node_index_from_data_id(data_index)
        self._set_search_color(node_index)
        return node_index

    def output_csv_from_node_ids(self, filename, node_ids=[], target=None):
        """Function of output csv file with node ids. This function is old version.

        Params:
            filename: The name of output csv file.

            node_ids: The array of node ids in output data.

            target: The array of values that is not input but use.
        """
        self.export(file_name=filename, node_ids=node_ids, target=target)

    def export(self, file_name, node_ids=[], target=None):
        """Function of output csv file with node ids.

        Params:
            file_name: The name of output csv file.

            node_ids: The array of node ids in output data.

            target: The array of values that is not input but use.
        """
        if target is None:
            d = self.number_data
        else:
            d = np.concatenate([self.number_data[node_ids], target.reshape(-1, 1)], axis=1)

        data = np.concatenate([d, self.text_data[node_ids]], axis=1)
        columns = np.concatenate([self.number_data_columns, self.text_data_columns])

        data = pd.DataFrame(data, columns=columns)
        data.to_csv(file_name, columns=columns, index=None)

    def color_point_cloud(self, target, color_type="rgb", normalize=True):
        """Function of coloring point cloud with target.

        Params:
            target: Array of coloring value.

            color_type: "rgb" or "gray".

            normalize: Normalize target data or not.
        """
        if color_type not in ["rgb", "gray"]:
            raise Exception("color_type {} is not usable.".format(color_type))

        if normalize:
            scaler = preprocessing.MinMaxScaler()
            self.normalized_target = scaler.fit_transform(np.array(target).reshape(-1, 1).astype(float))
        else:
            self.normalized_target = np.array(target)

        self.point_cloud_colors = self.normalized_target
        self.point_cloud_hex_colors = [""] * len(self.point_cloud)

        # calc color with painter's paint function.
        painter_resolver = PainterResolver()
        painter = painter_resolver.resolve(color_type)
        for i, t in enumerate(self.normalized_target):
            self.point_cloud_hex_colors[i] = painter.paint(t)

    def show_point_cloud(self, fig_size=(5, 5), node_size=5):
        """Function of showing point cloud.

        Params:
            fig_size: The size of figure.

            node_size: The size of node.
        """
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], c=self.point_cloud_hex_colors, s=node_size)
        plt.axis("off")
        plt.show()

    def save_point_cloud(self, file_name, fig_size=(5, 5), node_size=5):
        """Function of save point cloud.

        Params:
            file_name: Output figure name.

            fig_size: The size of figure.

            node_size: The size of node.
        """
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], c=self.point_cloud_hex_colors, s=node_size)
        plt.axis("off")
        plt.savefig(file_name)

    def _get_train_test_index(self, length, size=0.9):
        # get index of train, test data.
        threshold = int(length * size)
        index = np.random.permutation(length)
        train_index = np.sort(index[:threshold])
        test_index = np.sort(index[threshold:])
        return train_index, test_index

    def supervised_clustering_point_cloud(self, clusterer=None, target=None, train_size=0.9):
        """Function of supervised clustering of point cloud.

        Params:
            clusterer: Class of clustering.

            target: target data.

            train_size: The size of Training data.
        """
        painter_resolver = PainterResolver()
        painter = painter_resolver.resolve("rgb")

        if clusterer is not None and "fit" in dir(clusterer) and target is not None:
            # split train test.
            self.train_index, self.test_index = self._get_train_test_index(self.point_cloud.shape[0], train_size)
            x_train = self.number_data[self.train_index, :]
            x_test = self.number_data[self.test_index, :]
            y_train = target[self.train_index].astype(int)

            # fit & predict
            clusterer.fit(x_train, y_train.reshape(-1,))
            labels = np.zeros((self.point_cloud.shape[0], 1))
            labels[self.train_index] += y_train.reshape(-1, 1)
            labels[self.test_index] += clusterer.predict(x_test).reshape(-1, 1)

            # scaling predicted label & calc color.
            scaler = preprocessing.MinMaxScaler()
            labels = scaler.fit_transform(np.array(labels).reshape(-1, 1).astype(float))
            self.point_cloud_hex_colors = [painter.paint(i) for i in labels]

    def unsupervised_clustering_point_cloud(self, clusterer=None):
        """Function of unsupervised clustering of point cloud.

        Params:
            clusterer: Class of clustering.
        """
        painter_resolver = PainterResolver()
        painter = painter_resolver.resolve("rgb")

        if clusterer is not None and "fit" in dir(clusterer):
            clusterer.fit(self.number_data)

            # scaling clustered label & calc color.
            scaler = preprocessing.MinMaxScaler()
            labels = scaler.fit_transform(clusterer.labels_.reshape(-1, 1).astype(float))
            self.point_cloud_hex_colors = [painter.paint(i) if i >= 0 else "#000000" for i in labels]

    def _set_search_point_cloud_color(self, data_index):
        searched_color = ["#cccccc"] * len(self.point_cloud)
        for i in data_index:
            searched_color[i] = self.point_cloud_hex_colors[i]
        self.point_cloud_hex_colors = searched_color

    def search_point_cloud(self, search_dicts=None, target=None, search_type="and"):
        """Function of search point cloud with values.

        Params:
            search_dicts: The array of search options.

            target: The array of values that is not input but wan't to search.

            search_type: How to get column index in search_dicts. "column" or "index".
        """
        if search_type not in ["and", "or"]:
            raise Exception("search_type {} is not usable.".format(search_type))

        if target is None:
            d = self.number_data
        else:
            d = np.concatenate([self.number_data, target.reshape(-1, 1)], axis=1)

        data_index = self._get_searched_index(d, search_dicts, search_type)
        self._set_search_point_cloud_color(data_index)
        return data_index
