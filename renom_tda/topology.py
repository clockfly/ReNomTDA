# -*- coding: utf-8 -*.t-
from __future__ import print_function

import colorsys

import itertools

import matplotlib.pyplot as plt

import networkx as nx

import numpy as np

import scipy

from sklearn import cluster, decomposition, preprocessing


class Topology(object):
    """Topology Class

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.cluster import DBSCAN
        >>> from renom.tda.topology import Topology
        >>> from renom.tda.metric import Distance
        >>> from renom.tda.lens import L1Centrality, GaussianDensity, PCA
        >>> data, target = load_iris().data, load_iris().target
        >>> print(data.shape, target.shape)
        (150, 4) (150,)
        >>> print(data.shape)
        (150, 4)
        >>> topology = Topology()
        >>> metric = Distance(metric="euclidean")
        >>> lens = [L1Centrality(), GaussianDensity()]
        Mapping to L1Centrality and GaussianDensity.
        >>> topology.fit_transform(data, metric=metric, lens=lens)
        calculated distance matrix by Distance class using euclidean distance.
        projected by L1Centrality.
        projected by GaussianDensity.
        finish fit_transform.
        >>> topology.point_cloud.shape
        (150, 2)
        >>> clusterer = DBSCAN(eps=1, min_samples=3)
        >>> topology.map(resolution=10, overlap=0.5, clusterer=clusterer)
        mapping start, please wait...
        created 53 nodes.
        calculating cluster coordination.
        calculating edge.
        created 123 edges.
        >>> topology.nodes.shape
        (53, 2)
        >>> topology.edges.shape
        (123, 2)
        >>> len(topology.hypercubes)
        53
        >>> topology.color(target, dtype="categorical", ctype="rgb")
        Topology is colored by target value.
        >>> topology.show(fig_size=(10,10), node_size=5, edge_width=1, mode="spring", strength=0.05)
        Show topology usnig spring model.
    """

    def __init__(self, verbose=1):
        self.verbose = verbose
        self.metric = None
        self.lens = None
        self.scaler = None
        self.clusterer = None
        self.resolution = 0
        self.overlap = 0
        self.input_data = None
        self.point_cloud = None
        self.nodes = None
        self.node_sizes = None
        self.edges = None
        self.colors = None
        self.hypercubes = {}

    def fit_transform(self, data, metric=None, lens=None, scaler=preprocessing.MinMaxScaler()):
        """Function of projection data to point cloud.

        Params:
            data: raw data.

            metric: Class of trainsforming distance matrix.

            lens: List of projection lens class.

            scaler: Class of scaling.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        if data.ndim != 2:
            raise ValueError("Data must be 2d array.")

        self.input_data = data
        self.metric = metric
        self.lens = lens
        self.scaler = scaler

        # metricをもとに距離行列を作る。
        if (metric is not None) and ("fit_transform" in dir(metric)):
            if self.verbose == 1:
                print("calculated distance matrix by %s class using %s distance." %
                      (self.metric.__class__.__name__, self.metric.metric))
            dist_matrix = metric.fit_transform(data)
        else:
            # metricがNoneならdataをそのまま使う
            dist_matrix = data

        # lensによって射影後の次元数は異なるのでNoneで初期化
        self.point_cloud = None
        if (lens is not None) and (len(lens) > 0):
            for l in lens:
                if (l is not None) and "fit_transform" in dir(l):
                    if self.verbose == 1:
                        print("projected by %s." % (l.__class__.__name__))
                    if self.point_cloud is None:
                        self.point_cloud = l.fit_transform(dist_matrix)
                    else:
                        p = l.fit_transform(dist_matrix)
                        self.point_cloud = np.concatenate([self.point_cloud, p], axis=1)
                else:
                    self.point_cloud = dist_matrix
        else:
            # lensがNoneならdist_matrixをそのまま使う
            self.point_cloud = dist_matrix

        # scalerで正規化。
        if self.point_cloud is not None:
            if (scaler is not None) and ("fit_transform" in dir(scaler)):
                self.point_cloud = scaler.fit_transform(self.point_cloud)

        if self.verbose == 1:
            print("finish fit_transform.")

    def map(self, resolution=10, overlap=0.5, clusterer=cluster.DBSCAN(eps=1, min_samples=1)):
        """Function of mapping point cloud to topological space.

        Params:
            resolution: The number of divisoin of each axis. It controls the number of division.

            overlap: The amount of overlap of each division. It represents easiness of connection.

            clusterer: The method of clustering of each division.
        """
        if self.point_cloud is None:
            raise Exception("Point cloud is not exist yet.")

        if self.point_cloud.ndim != 2:
            raise ValueError("Data must be 2d array.")

        if resolution <= 0:
            raise ValueError("Resolution must be greater than Zero.")

        if overlap < 0:
            raise ValueError("Overlap must be greater than Zero.")

        self.clusterer = clusterer
        self.resolution = resolution
        self.overlap = overlap
        self.hypercubes = {}

        scaler = preprocessing.MinMaxScaler()
        scaled_data = scaler.fit_transform(self.input_data)

        # 元のデータ点のindex。hypercubeに含まれるデータを特定するのに使う。
        input_data_indexs = np.arange(self.point_cloud.shape[0])

        # 作成したノードの数。hypercubeのidに使う。
        created_node_count = 0

        # 分割する間隔
        chunk_width = 1. / resolution

        # 重なり
        overlap_width = chunk_width * overlap

        if self.verbose == 1:
            print("mapping start, please wait...")

        # hypercubeに分割する
        # resolution: 10, point_cloudが2次元なら、10x10の100個のhypercubeを作る
        for i in itertools.product(np.arange(resolution), repeat=self.point_cloud.shape[1]):
            # hypercubeの上限、下限を設定する
            floor = np.array(i) * chunk_width - overlap_width
            roof = np.array([n + 1 for n in i]) * chunk_width + overlap_width

            # hypercubeの上限、下限からマスクを求める
            mask_floor = self.point_cloud > floor
            mask_roof = self.point_cloud < roof
            mask = np.all(mask_floor & mask_roof, axis=1)

            # hypercubeに含まれるidを求める。
            masked_data_indexs = input_data_indexs[mask]

            # hypercubeに含まれる元の次元のデータを求める。
            masked_data = scaled_data[mask]

            if masked_data.shape[0] > 0:
                if (clusterer is not None) and ("fit" in dir(clusterer)):
                    # hypercube内の点を元の次元で再度クラスタリング
                    clusterer.fit(masked_data)

                    # クラスタリング結果のラベルでループを回す
                    for label in np.unique(clusterer.labels_):
                        # ラベルが-1の点はノイズとする
                        if label != -1:
                            hypercube_value = list(masked_data_indexs[clusterer.labels_ == label])
                            # 同じ値を持つhypercubeは作らない
                            if hypercube_value not in self.hypercubes.values():
                                self.hypercubes.update(
                                    {created_node_count: hypercube_value})
                                created_node_count += 1
                else:
                    self.hypercubes.update({created_node_count: masked_data_indexs})
                    created_node_count += 1

        if self.verbose == 1:
            print("created %s nodes." % (len(self.hypercubes)))

        # nodeが0個なら終了
        if len(self.hypercubes) == 0:
            return

        if self.verbose == 1:
            print("calculating cluster coordination.")

        # ノードの配列を(ノード数, 投影する次元数)の大きさで初期化する
        self.nodes = np.zeros((len(self.hypercubes), self.point_cloud.shape[1]))
        self.node_sizes = np.zeros((len(self.hypercubes), 1))

        # hypercubeのキーでループ
        for key in self.hypercubes.keys():
            # ノード内のデータを取得
            data_in_node = self.point_cloud[self.hypercubes[key]]
            # ノードの座標をノード内の点の座標の平均で求める
            node_coordinate = np.average(data_in_node, axis=0)
            # ノードの配列を更新する
            self.nodes[int(key)] += node_coordinate
            self.node_sizes[int(key)] += len(data_in_node)

        if self.verbose == 1:
            print("calculating edge.")

        # エッジの配列を初期化（最終的な大きさはわからないのでNoneで初期化）
        self.edges = None

        # hypercubeのキーの組み合わせでループ
        for keys in itertools.combinations(self.hypercubes.keys(), 2):
            cube1 = set(self.hypercubes[keys[0]])
            cube2 = set(self.hypercubes[keys[1]])

            # cube1とcube2で重なる点があったらエッジを作る
            if len(cube1.intersection(cube2)) > 0:
                edge = np.array(keys).reshape(1, -1)
                if self.edges is None:
                    self.edges = edge
                else:
                    self.edges = np.concatenate([self.edges, edge], axis=0)

        if self.verbose == 1:
            if self.edges is None:
                print("created 0 edges.")
            else:
                print("created %s edges." % (len(self.edges)))

    def _get_hex_color(self, i):
        """
        rgbの色コードを返す関数。
        """
        # hsv色空間でiが0~1を青~赤に対応させる。
        c = colorsys.hsv_to_rgb((1 - i) * 240 / 360, 1.0, 0.7)
        return "#%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

    def _get_grayscale_color(self, i):
        """
        グレースケールの色コードを返す関数。
        """
        i = 1 - i
        return "#%02x%02x%02x" % (int((i + 0.1) * 200), int((i + 0.1) * 200), int((i + 0.1) * 200))

    def _rescale_target(self, target):
        ret_target = target

        # 統計量を求める
        q1 = scipy.stats.scoreatpercentile(target, 25)
        median = np.median(target)
        q3 = scipy.stats.scoreatpercentile(target, 75)

        # スケーリングの設定
        scaler_min_q1 = preprocessing.MinMaxScaler(feature_range=(0, 0.25))
        scaler_q1_median = preprocessing.MinMaxScaler(feature_range=(0.25, 0.5))
        scaler_median_q3 = preprocessing.MinMaxScaler(feature_range=(0.5, 0.75))
        scaler_q3_max = preprocessing.MinMaxScaler(feature_range=(0.75, 1))

        # indexを取得
        index_min_q1 = np.where(target <= q1)[0]
        index_q1_median = np.where(((target >= q1) & (target <= median)))[0]
        index_median_q3 = np.where(((target >= median) & (target <= q3)))[0]
        index_q3_max = np.where(target >= q3)[0]

        # スケーリングを実行
        target_min_q1 = scaler_min_q1.fit_transform(target[index_min_q1])
        target_q1_median = scaler_q1_median.fit_transform(target[index_q1_median])
        target_median_q3 = scaler_median_q3.fit_transform(target[index_median_q3])
        target_q3_max = scaler_q3_max.fit_transform(target[index_q3_max])

        # returnする色のarrayに代入する
        ret_target[index_min_q1] = target_min_q1
        ret_target[index_q1_median] = target_q1_median
        ret_target[index_median_q3] = target_median_q3
        ret_target[index_q3_max] = target_q3_max
        return np.array(ret_target)

    def color(self, target, dtype="numerical", ctype="rgb", feature_range=(0, 1), normalized=False):
        """Function of coloring topology.

        Params:
            target: Array of coloring data.

            dtype: The type of data. If dtype is "numerical", node is colored by mean value.
                   If dtype is "categorical", node is colored by mode value.

            ctype: The type of node color. RGB or Grayscale.

            feature_range: The range of color gradation.

            normalized: Target value is normalized or not. If target is aleady normalized, this variable is True.
        """
        if target is None:
            raise Exception("Target must not None.")

        if type(target) is not np.ndarray:
            target = np.array(target)

        if len(target) != self.input_data.shape[0]:
            raise ValueError("Target data size must be same with input data")

        if dtype not in ["numerical", "categorical"]:
            raise ValueError("Data type must be 'numerical' or 'categorical'.")

        if ctype not in ["rgb", "gray"]:
            raise ValueError("Color type must be 'rgb' or 'gray'.")

        if len(target.shape) == 1:
            target = target.reshape(-1, 1)

        self.colors = [""] * len(self.hypercubes)

        # targetを0~1に正規化する
        target = target.astype(np.float)
        if normalized:
            scaled_target = target
        else:
            scaler = preprocessing.MinMaxScaler()
            scaled_target = scaler.fit_transform(target)

        color_num_list = np.zeros((len(self.hypercubes), 1))
        for key in self.hypercubes.keys():
            target_in_node = scaled_target[self.hypercubes[key]]
            if dtype == "categorical":
                # カテゴリデータは最頻値
                color_num_list[int(key)] = scipy.stats.mode(target_in_node)[0][0]
            elif dtype == "numerical":
                # 数値データは平均
                color_num_list[int(key)] = np.average(target_in_node)

        color_num_list = self._rescale_target(color_num_list)
        for key in self.hypercubes.keys():
            if ctype == "rgb":
                hex_color = self._get_hex_color(color_num_list[int(key)])
            elif ctype == "gray":
                hex_color = self._get_grayscale_color(color_num_list[int(key)])

            self.colors[int(key)] = hex_color

    def show(self, fig_size=(5, 5), node_size=10, edge_width=2, mode=None, strength=None):
        """Function of visualize topology.

        Params:
            fig_size: The size of output image.

            node_size: The size of nodes.

            edge_width: The width of edges.

            mode: Visualization mode. None or "spring".
                  If this is "spring", node coordination is calculated by spring model.

            strength: The strength of spring in spring model.
        """

        # node_sizeを計算する。最大の大きさを決める。
        max_node_size = fig_size[0]
        self.node_sizes[self.node_sizes > max_node_size] = max_node_size
        draw_node_sizes = self.node_sizes * fig_size[0] * 2

        if mode == "spring":
            # グラフの初期位置を設定する。
            init_position = {}

            # ノードの座標が1次元の時はPCAで2次元に落とした座標をクラスタの初期位置とする。
            if self.nodes.shape[1] == 1:
                m = decomposition.PCA(n_components=2)
                s = preprocessing.MinMaxScaler()
                d = m.fit_transform(self.input_data)
                d = s.fit_transform(d)
                for k in self.hypercubes.keys():
                    data_in_node = d[self.hypercubes[k]]
                    init_position.update({int(k): np.average(data_in_node, axis=0)})
            elif self.nodes.shape[1] == 2 or self.nodes.shape[1] == 3:
                # 2次元,3次元の時はノードの座標値上位２つを初期値にする。
                for k in self.hypercubes.keys():
                    init_position.update({int(k): self.nodes[int(k), :2]})

            # ネットワークグラフで描画する。
            fig = plt.figure(figsize=fig_size)
            graph = nx.Graph(pos=init_position)
            graph.add_nodes_from(range(len(self.nodes)))
            graph.add_edges_from(self.edges)
            init_position = nx.spring_layout(graph, pos=init_position, k=strength)
            nx.draw_networkx(graph, pos=init_position,
                             node_size=draw_node_sizes, node_color=self.colors,
                             width=edge_width,
                             edge_color=[self.colors[e[0]] for e in self.edges],
                             with_labels=False)
        else:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(111)
            for e in self.edges:
                ax.plot([self.nodes[e[0], 0], self.nodes[e[1], 0]],
                        [self.nodes[e[0], 1], self.nodes[e[1], 1]],
                        c=self.colors[e[0]])
            ax.scatter(self.nodes[:, 0], self.nodes[:, 1], c=self.colors, s=draw_node_sizes)
        plt.axis("off")
        plt.show()


class SearchableTopology(Topology):
    """SearchableTopology Class

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.cluster import DBSCAN
        >>> from renom.tda.topology import SearchableTopology
        >>> from renom.tda.metric import Distance
        >>> from renom.tda.lens import PCA
        >>> data, target = load_iris().data, load_iris().target
        >>> topology = SearchableTopology()
        >>> metric = Distance(metric="euclidean")
        >>> lens = [PCA(components=[0,1])]
        >>> topology.fit_transform(data, metric=metric, lens=lens)
        calculated distance matrix by Distance class using euclidean distance.
        projected by PCA.
        finish fit_transform.
        >>> clusterer = DBSCAN(eps=1, min_samples=3)
        >>> topology.map(resolution=10, overlap=0.5, clusterer=clusterer)
        mapping start, please wait...
        created 28 nodes.
        calculating cluster coordination.
        calculating edge.
        created 49 edges.
        >>> label_setosa = ["setosa"]*50
        >>> label_versicolor = ["versicolor"]*50
        >>> label_virginica = ["virginica"]*50
        >>> labels = label_setosa + label_versicolor + label_virginica
        >>> len(labels)
        150
        >>> topology.regist_categorical_data(labels)
        >>> topology.categorical_data.shape
        (150, 1)
        >>> topology.color(target, dtype="categorical", ctype="rgb")
        >>> topology.search("setosa")
        setosa is in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                      41, 42, 43, 44, 45, 46, 47, 48, 49] data.
        setosa is in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] node.
        >>> topology.show(fig_size=(5,5), node_size=5, edge_width=1)
    """

    def __init__(self, verbose=1):
        super(SearchableTopology, self).__init__(verbose)
        self.categorical_data = None

    def regist_categorical_data(self, data):
        """Function of regist categorical data for search.

        Params:
            data: Categorical data.
        """
        if type(data) is not np.ndarray:
            data = np.array(data)

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        self.categorical_data = data

    def search(self, value):
        """Function of searching categorical data.

        Params:
            value: Search value.
        """
        # 全ての色をグレーにする
        searched_color = ["#cccccc"] * len(self.hypercubes.keys())
        index = []
        for i in range(self.categorical_data.shape[0]):
            for j in range(self.categorical_data.shape[1]):
                if value in self.categorical_data[i, j]:
                    index.append(i)

        if self.verbose == 1:
            print("%s is in %s data." % (value, index))
        values = self.hypercubes.values()

        node_index = []
        for i, val in enumerate(values):
            s1 = set(val)
            s2 = set(index)
            if len(s1.intersection(s2)) > 0:
                node_index.append(i)
        if self.verbose == 1:
            print("%s is in %s node." % (value, node_index))

        for i in node_index:
            searched_color[i] = self.colors[i]
        self.colors = searched_color

    def advanced_search(self, search_dict, target):
        searched_color = ["#cccccc"] * len(self.hypercubes.keys())
        index = []

        for key in search_dict.keys():
            # 数値データの検索
            if search_dict[key]["data_type"] == 0:
                if search_dict[key]["index"] < self.input_data.shape[1]:
                    search_data = self.input_data[:, search_dict[key]["index"]]
                else:
                    search_data = (target - target.mean()) / (target.std() + 1e-6)
                if search_dict[key]["operator"] == "=":
                    index.extend(np.where(search_data == search_dict[key]["value"])[0])
                elif search_dict[key]["operator"] == ">":
                    index.extend(np.where(search_data > search_dict[key]["value"])[0])
                elif search_dict[key]["operator"] == "<":
                    index.extend(np.where(search_data < search_dict[key]["value"])[0])
            # 文字列データの検索
            elif search_dict[key]["data_type"] == 1:
                search_data = self.categorical_data[:, search_dict[key]["index"]]
                if search_dict[key]["operator"] == "=":
                    index.extend(np.where(search_data == search_dict[key]["value"])[0])
                elif search_dict[key]["operator"] == "like":
                    for i in range(search_data.shape[0]):
                        if search_dict[key]["value"] in search_data[i]:
                            index.append(i)

        values = self.hypercubes.values()
        node_index = []
        for i, val in enumerate(values):
            s1 = set(val)
            s2 = set(index)
            if len(s1.intersection(s2)) > 0:
                node_index.append(i)

        for i in node_index:
            searched_color[i] = self.colors[i]
        self.colors = searched_color

    def get_hypercube(self, key):
        """Function of getting value of hypercubes.

        Return:
           Categorical data and data in key hypercube.

        Params:
            key: The ID of hypercube.
        """
        return self.categorical_data[self.hypercubes[int(key)]], self.input_data[self.hypercubes[int(key)]]
