# -*- coding: utf-8 -*.t-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster, preprocessing

from renom_tda.lens import Lenses
from renom_tda.utils import DistUtil, MapUtil, GraphUtil
from renom_tda.painter import PainterResolver
from renom_tda.presenter import PresenterResolver
from renom_tda.searcher import SearcherResolver


class Topology(object):
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
        self.number_data_avg = None
        self.number_data_std = None
        # transform
        self.dist_util = None
        self.metric = None
        self.lens = None
        self.scaler = preprocessing.StandardScaler()
        # map
        self.resolution = 0
        self.overlap = 0
        self.eps = 0
        self.min_samples = 0
        # output data
        self.graph = None
        self.train_index = np.array([])
        self.point_cloud = None
        self.hypercubes = {}
        self.nodes = None
        self.edges = None
        self.node_sizes = None
        self.colors = None
        self.color_target = None
        self.hex_colors = None

    def load_data(self, number_data, text_data=None, text_data_columns=None,
                  number_data_columns=None, standardize=False):
        """Function of load data to this instance.

        Params:
            number_data: Data using calclate topology.

            text_data: Text data correspond to number data.

            text_data_columns: Column names of text data.

            number_data_columns: Column names of number data.

            standardize: standardize number data or not.
        """
        # number dataは必須
        if number_data is None:
            raise ValueError("Number data must not None.")

        # number_dataをnumpy配列に直す
        if number_data is not None and type(number_data) is not np.ndarray:
            number_data = np.array(number_data)

        # number_dataが二次元配列以外の時errorを発生させる
        if number_data.ndim != 2:
            raise ValueError("Number data must be 2d array.")

        # text_dataをnumpy配列に直す
        if text_data is not None and type(text_data) is not np.ndarray:
            text_data = np.array(text_data)

        # text_data_columnsがある時、text_dataは必須
        if text_data_columns is not None and text_data is None:
            raise ValueError("When you input text data columns, text data must be exist.")

        if text_data_columns is not None and type(text_data_columns) is not np.ndarray:
            text_data_columns = np.array(text_data_columns)

        # text_data_columnsとtext_dataは列数が等しい必要がある
        if text_data_columns is not None and text_data is not None:
            if text_data.shape[1] != text_data_columns.shape[0]:
                raise ValueError("Text data and text data columns must have same number of columns.")

        if number_data_columns is not None and type(number_data_columns) is not np.ndarray:
            number_data_columns = np.array(number_data_columns)

        self.standardize = standardize

        self.number_data = number_data
        self.number_data_columns = number_data_columns
        self.text_data = text_data
        self.text_data_columns = text_data_columns

        if standardize:
            self.std_number_data = self.scaler.fit_transform(self.number_data)
        else:
            self.std_number_data = number_data

    def load(self, loader, standardize=True):
        self.standardize = standardize
        self.number_data, self.text_data, self.number_data_columns, self.text_data_columns = loader.load()
        if standardize:
            self.std_number_data = self.scaler.fit_transform(self.number_data)
        else:
            self.std_number_data = np.array(self.number_data)

    def fit_transform(self, metric=None, lens=None, scaler=preprocessing.MinMaxScaler()):
        self.metric = metric
        self.lens = lens
        self.scaler = scaler

        # dist matrix
        if metric is None:
            d = self.std_number_data
        else:
            self.dist_util = DistUtil(metric=metric)
            d = self.dist_util.calc_dist_matrix(self.std_number_data)

        # transform
        if lens is not None:
            self.lenses = Lenses(filters=lens)
            d = self.lenses.fit_transform(d)

        if (scaler is not None) and ("fit_transform" in dir(scaler)):
            d = scaler.fit_transform(d)

        self.point_cloud = d

    def map(self, resolution=10, overlap=1, eps=1, min_samples=1):
        self.resolution = resolution
        self.overlap = overlap
        self.eps = eps
        self.min_samples = min_samples

        if self.dist_util is None:
            self.dist_util = DistUtil()
            self.dist_util.calc_dist_matrix(self.std_number_data)
        eps_ = self.dist_util.calc_eps(eps)

        clusterer = cluster.DBSCAN(eps=eps_, min_samples=min_samples)
        self.mapper = MapUtil(resolution=resolution, overlap=overlap, clusterer=clusterer)
        self.hypercubes = self.mapper.map(self.std_number_data, self.point_cloud)

        self.graph = GraphUtil(point_cloud=self.point_cloud, hypercubes=self.hypercubes)
        self.nodes, self.node_sizes = self.graph.calc_node_coordinate()
        self.edges = self.graph.calc_edges()

    def color(self, target, color_method="mean", color_type="rgb", normalize=True):
        if normalize:
            scaler = preprocessing.MinMaxScaler()
            self.normalized_target = scaler.fit_transform(np.array(target).reshape(-1, 1).astype(float))
        else:
            self.normalized_target = np.array(target)

        self.colors, self.hex_colors = self.graph.color(self.normalized_target, color_method, color_type)

    def _get_presenter(self, fig_size, node_size, edge_width, mode, strength):
        resolver = PresenterResolver(fig_size, node_size, edge_width, strength)
        return resolver.resolve(mode)

    def show(self, fig_size=(5, 5), node_size=5, edge_width=1, mode="normal", strength=None):
        presenter = self._get_presenter(fig_size, node_size, edge_width, mode, strength)
        presenter.show(self.nodes, self.edges, self.node_sizes, self.hex_colors)

    def save(self, filename, fig_size=(5, 5), node_size=5, edge_width=1, mode="normal", strength=None):
        presenter = self._get_presenter(fig_size, node_size, edge_width, mode, strength)
        presenter.save(self.nodes, self.edges, self.node_sizes, self.hex_colors)

    def _node_index_from_data_id(self, data_index):
        # データのインデックスからそのデータを含むノードのインデックスを返す
        node_index = []
        values = self.hypercubes.values()
        for i, val in enumerate(values):
            s1 = set(val)
            s2 = set(data_index)
            if len(s1.intersection(s2)) > 0:
                node_index.append(i)
        return node_index

    def _set_search_color(self, node_index):
        # 検索結果の色をセットする
        # 検索結果以外のノードをグレーにする
        searched_color = ["#cccccc"] * len(self.hypercubes.keys())
        for i in node_index:
            searched_color[i] = self.hex_colors[i]
        self.hex_colors = searched_color

    def search_from_values(self, search_dicts=None, target=None, search_type="and"):
        self.search(search_dicts=search_dicts, target=target, search_type=search_type)

    def _get_searched_index(self, data, search_dicts, search_type="and"):
        resolver = SearcherResolver(data, self.text_data, self.number_data_columns, self.text_data_columns)

        data_index = []

        for d in search_dicts:
            searcher = resolver.resolve(d["data_type"])
            index = searcher.search(column=d["column"], operator=d["operator"], value=d["value"])

            if len(data_index) > 0:
                # concatenate
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
        if target is None:
            d = self.number_data
        else:
            d = np.concatenate([self.number_data, target.reshape(-1, 1)], axis=1)

        data_index = self._get_searched_index(d, search_dicts, search_type)
        node_index = self._node_index_from_data_id(data_index)
        self._set_search_color(node_index)
        return node_index

    def output_csv_from_node_ids(self, filename, node_ids=[], target=None):
        """Function of output csv file with node ids.

        Params:
            filename: The name of output csv file.

            node_ids: The array of node ids in output data.

            target: The array of values that is not input but use.

            skip_header: Skip header or not. If you set False, text_column_names and number_column_names must not None.
        """
        self.export(file_name=filename, node_ids=node_ids, target=target)

    def export(self, file_name, node_ids=[], target=None):
        if target is None:
            d = self.number_data
        else:
            d = np.concatenate([self.number_data[node_ids], target.reshape(-1, 1)], axis=1)

        data = np.concatenate([d, self.text_data[node_ids]], axis=1)
        columns = np.concatenate([self.number_data_columns, self.text_data_columns])

        data = pd.DataFrame(data, columns=columns)
        data.to_csv(file_name, columns=columns, index=None)

    def color_point_cloud(self, target, color_type="rgb", normalize=False):
        if normalize:
            scaler = preprocessing.MinMaxScaler()
            self.normalized_target = scaler.fit_transform(np.array(target).reshape(-1, 1).astype(float))
        else:
            self.normalized_target = np.array(target)

        self.point_cloud_colors = self.normalized_target
        self.point_cloud_hex_colors = [""] * len(self.point_cloud)

        painter_resolver = PainterResolver()
        painter = painter_resolver.resolve(color_type)
        for i, t in enumerate(target):
            self.point_cloud_hex_colors[i] = painter.paint(t)

    def show_point_cloud(self, fig_size=(5, 5), node_size=5):
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], c=self.point_cloud_hex_colors, s=node_size)
        plt.axis("off")
        plt.show()

    def save_point_cloud(self, file_name, fig_size=(5, 5), node_size=5):
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], c=self.point_cloud_hex_colors, s=node_size)
        plt.axis("off")
        plt.savefig(file_name)

    def _get_train_test_index(self, length, size=0.9):
        # 学習データとテストデータに分ける
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
            # 教師データとテストデータに分ける
            self.train_index, self.test_index = self._get_train_test_index(self.point_cloud.shape[0], train_size)
            x_train = self.number_data[self.train_index, :]
            x_test = self.number_data[self.test_index, :]
            y_train = target[self.train_index].astype(int)

            # 目的変数がラベルデータ(int)なら分類&predictする
            clusterer.fit(x_train, y_train)
            labels = np.zeros((self.point_cloud.shape[0], 1))
            labels[self.train_index] += y_train.reshape(-1, 1)
            labels[self.test_index] += clusterer.predict(x_test).reshape(-1, 1)

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
            scaler = preprocessing.MinMaxScaler()
            labels = scaler.fit_transform(clusterer.labels_.reshape(-1, 1).astype(float))
            self.point_cloud_hex_colors = [painter.paint(i) if i >= 0 else "#000000" for i in labels]

    def _set_search_point_cloud_color(self, data_index):
        # 検索結果以外の色をグレーにする
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
        if target is None:
            d = self.number_data
        else:
            d = np.concatenate([self.number_data, target.reshape(-1, 1)], axis=1)

        data_index = self._get_searched_index(d, search_dicts, search_type)
        self._set_search_point_cloud_color(data_index)
        return data_index


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from loader import CSVLoader
    from lens import L1Centrality, GaussianDensity

    iris = load_iris()
    data = iris.data
    target = iris.target

    t = Topology()
    # loader = CSVLoader("test/data/test.csv")
    # t.load(loader=loader, standardize=True)
    t.load_data(data)

    lens = [L1Centrality(), GaussianDensity()]
    t.fit_transform(metric="euclidean", lens=lens)

    t.map(resolution=10, overlap=1, eps=1, min_samples=1)

    # target = [0., 1.]
    t.color(target, color_method="mean", color_type="rgb", normalize=True)

    search_dicts = [{
        "data_type": "number",
        "operator": ">",
        "column": 0,
        "value": 5.0,
    }, {
        "data_type": "number",
        "operator": "=",
        "column": -1,
        "value": 1.0,
    }]
    searched_node_index = t.search_from_values(search_dicts=search_dicts, target=target, search_type="and")
    # t.export(file_name='test.csv', node_ids=searched_node_index)
    t.show()

    # t.color_point_cloud(target=target)
    # t.search_point_cloud(search_dicts=search_dicts, target=target, search_type="or")
    # t.show_point_cloud()
