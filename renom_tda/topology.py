# -*- coding: utf-8 -*.t-
from __future__ import print_function

import colorsys

import itertools

import matplotlib.pyplot as plt

import networkx as nx

import numpy as np

import pandas as pd

import scipy

from sklearn import cluster, decomposition, preprocessing


class TopologyCore(object):
    """
    TDA Core class
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
        self.number_data_avg = None
        self.number_data_std = None
        # transform
        self.metric = None
        self.lens = None
        self.scaler = None
        # map
        self.resolution = 0
        self.overlap = 0
        self.eps = 0
        self.min_samples = 0
        # output data
        self.train_index = np.array([])
        self.point_cloud = None
        self.hypercubes = {}
        self.nodes = None
        self.edges = None
        self.node_sizes = None
        self.colors = None
        self.color_target = None
        self.hex_colors = None

    def _standardize(self, data):
        self.number_data_avg = np.average(data, axis=0)
        self.number_data_std = np.std(data, axis=0)
        standardize_data = (data - self.number_data_avg) / (self.number_data_std + 1e-10)
        return standardize_data

    def _re_standardize(self, data):
        if self.standardize:
            standardize_data = data * (self.number_data_std + 1e-10) + self.number_data_avg
            return standardize_data
        else:
            return data

    def _normalize(self, data, feature_range=(0, 1)):
        data = data.astype(np.float)
        scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
        return scaler.fit_transform(data)

    def _create_dist_matrix(self, metric):
        # metricをもとに距離行列を作る。
        if (metric is not None) and ("fit_transform" in dir(metric)):
            if self.verbose == 1:
                print("calculated distance matrix by %s class using %s distance." %
                      (self.metric.__class__.__name__, self.metric.metric))
            return metric.fit_transform(self.number_data)
        else:
            # metricがNoneならdataをそのまま返す
            return self.number_data

    def _project_data(self, data, lens):
        ret_data = None
        # lensによって射影後の次元数は異なるのでNoneで初期化
        if (lens is not None) and (len(lens) > 0):
            for l in lens:
                if (l is not None) and "fit_transform" in dir(l):
                    if self.verbose == 1:
                        print("projected by %s." % (l.__class__.__name__))

                    if ret_data is None:
                        ret_data = l.fit_transform(data)
                    else:
                        p = l.fit_transform(data)
                        ret_data = np.concatenate([ret_data, p], axis=1)
            return ret_data
        else:
            return data

    def _scale_data(self, data, scaler):
        if (scaler is not None) and ("fit_transform" in dir(scaler)):
            return scaler.fit_transform(data)
        else:
            return data

    def _calc_dist_vec(self, data):
        dist_mat = scipy.spatial.distance.cdist(data, data)
        dist_vec = np.trim_zeros(np.unique(dist_mat))
        return dist_vec

    def _calc_dist_from_eps(self, dist_vec, eps):
        d_min = dist_vec.min()
        d_max = dist_vec.max()
        return d_min + (d_max - d_min) * eps

    def _create_hypercubes(self, resolution, overlap, clusterer):
        # point_cloudのデータ点のindex。hypercubeに含まれるデータを特定するのに使う。
        input_data_ids = np.arange(self.point_cloud.shape[0])

        # 作成したノードの数。hypercubeのidに使う。
        created_node_count = 0

        # 分割する間隔
        chunk_width = 1. / resolution
        # 重なり幅
        overlap_width = chunk_width * overlap

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
            masked_data_ids = input_data_ids[mask]

            # hypercubeに含まれる元の次元のデータを求める。
            masked_data = self.number_data[mask]

            # hypercube内の点を元の次元の距離で再度クラスタリング
            if masked_data.shape[0] > 0:
                if (clusterer is not None) and ("fit" in dir(clusterer)):
                    clusterer.fit(masked_data)

                    # クラスタリング結果のラベルでループを回す
                    for label in np.unique(clusterer.labels_):
                        # ラベルが-1の点はノイズとする
                        if label != -1:
                            hypercube_value = list(masked_data_ids[clusterer.labels_ == label])
                            # 同じ値を持つhypercubeは作らない
                            if hypercube_value not in self.hypercubes.values():
                                self.hypercubes.update(
                                    {created_node_count: hypercube_value})
                                created_node_count += 1
                else:
                    self.hypercubes.update({created_node_count: masked_data_ids})
                    created_node_count += 1

    def _calc_node_coordinate(self):
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

    def _calc_edges(self):
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

    def _calc_color_values(self, target, color_method):
        ret_colors = np.zeros((len(self.hypercubes), 1))
        for key in self.hypercubes.keys():
            target_in_node = target[self.hypercubes[key]]
            if color_method == "mode":
                # 最頻値
                ret_colors[int(key)] = scipy.stats.mode(target_in_node)[0][0]
            elif color_method == "mean":
                # 平均値
                ret_colors[int(key)] = np.average(target_in_node)
        return ret_colors

    def _rescale_colors(self, colors, thresholds=[0, 0.25, 0.5, 0.75, 1.0]):
        ret_colors = np.zeros((len(colors), 1))
        for index, threshold in enumerate(thresholds):
            if index == len(thresholds) - 1:
                break
            else:
                r_max = thresholds[index + 1]
            r_min = threshold

            scaler = preprocessing.MinMaxScaler(feature_range=(r_min, r_max))

            q_min = scipy.stats.scoreatpercentile(colors, r_min * 100)
            q_max = scipy.stats.scoreatpercentile(colors, r_max * 100)

            index_q = np.where(((q_min <= colors) & (colors <= q_max)))[0]
            target_q = scaler.fit_transform(colors[index_q])
            ret_colors[index_q] = target_q
        return ret_colors

    def _hex_color(self, i):
        # rgbの色コードを返す関数。
        # hsv色空間でiが0~1を青~赤に対応させる。
        c = colorsys.hsv_to_rgb((1 - i) * 240 / 360, 1.0, 0.7)
        return "#%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

    def _grayscale_color(self, i):
        # グレースケールの色コードを返す関数。
        i = 1 - i
        return "#%02x%02x%02x" % (int((i + 0.1) * 200), int((i + 0.1) * 200), int((i + 0.1) * 200))

    def _calc_hex_color_values(self, color_type):
        for key in self.hypercubes.keys():
            if color_type == "rgb":
                hex_color = self._hex_color(self.colors[int(key)])
            elif color_type == "gray":
                hex_color = self._grayscale_color(self.colors[int(key)])

            self.hex_colors[int(key)] = hex_color

    def _plot(self, fig_size, node_size, edge_width):
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        for e in self.edges:
            ax.plot([self.nodes[e[0], 0], self.nodes[e[1], 0]],
                    [self.nodes[e[0], 1], self.nodes[e[1], 1]],
                    c=self.hex_colors[e[0]], lw=edge_width, zorder=1)
        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], c=self.hex_colors, s=self.node_sizes * node_size, zorder=2)
        plt.axis("off")

    def _calc_init_position(self):
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
        return init_position

    def _spring_plot(self, fig_size, node_size, edge_width, strength):
        init_position = self._calc_init_position()

        # ネットワークグラフで描画する。
        fig = plt.figure(figsize=fig_size)
        graph = nx.Graph(pos=init_position)
        graph.add_nodes_from(range(len(self.nodes)))
        graph.add_edges_from(self.edges)
        position = nx.spring_layout(graph, pos=init_position, k=strength)

        nx.draw_networkx(graph, pos=position,
                         node_size=self.node_sizes * node_size, node_color=self.hex_colors,
                         width=edge_width,
                         edge_color=[self.hex_colors[e[0]] for e in self.edges],
                         with_labels=False)
        plt.axis("off")

    def _get_search_column_index(self, columns, search_dict, search_type):
        column_index = None
        if search_type == "column":
            index_array = np.where(columns == search_dict["column"])[0]
            if(len(index_array) > 0):
                column_index = index_array[0]
        else:
            column_index = search_dict["column"]
        return column_index

    def _number_search(self, data, value, operator):
        index = []
        if operator == "=":
            index.extend(np.where(data == value)[0])
        elif operator == ">":
            index.extend(np.where(data >= value)[0])
        if operator == "<":
            index.extend(np.where(data <= value)[0])
        return index

    def _text_search(self, data, value, operator):
        index = []
        if operator == "=":
            index.extend(np.where(data == value)[0])
        elif operator == "like":
            for i, d in enumerate(data):
                if value in d:
                    index.append(i)
        return index

    def _data_index_from_search_dict(self, search_number_data, search_number_data_columns, search_dicts, search_type):
        data_index = []
        for search_dict in search_dicts:
            index = []
            if search_dict["data_type"] == "number":
                column_index = self._get_search_column_index(search_number_data_columns, search_dict, search_type)
                if column_index is not None:
                    index = self._number_search(search_number_data[:, column_index], search_dict["value"], search_dict["operator"])
            else:
                column_index = self._get_search_column_index(self.text_data_columns, search_dict, search_type)
                if column_index is not None:
                    index = self._text_search(self.text_data[:, column_index], search_dict["value"], search_dict["operator"])

            if len(data_index) > 0:
                if len(index) > 0:
                    s1 = set(data_index)
                    s2 = set(index)
                    data_index = list(s1.intersection(s2))
            else:
                data_index = index

        return data_index

    def _node_index_from_data_id(self, data_index):
        node_index = []
        values = self.hypercubes.values()
        for i, val in enumerate(values):
            s1 = set(val)
            s2 = set(data_index)
            if len(s1.intersection(s2)) > 0:
                node_index.append(i)
        return node_index

    def _set_search_color(self, node_index):
        searched_color = ["#cccccc"] * len(self.hypercubes.keys())
        for i in node_index:
            searched_color[i] = self.hex_colors[i]
        self.hex_colors = searched_color

    def _concatenate_target(self, data, target):
        if target is not None:
            number_data = np.concatenate([data, target.reshape(-1, 1)], axis=1)
            if self.number_data_columns is not None:
                number_data_columns = np.concatenate([self.number_data_columns, np.array(["target"])], axis=0)
            else:
                number_data_columns = None
            return number_data, number_data_columns
        else:
            return data, self.number_data_columns

    def _data_index_from_node_id(self, node_ids):
        data_index = []
        for nid in node_ids:
            data_index.extend(self.hypercubes[nid])
        return list(set(data_index))

    def load_data(self, number_data, text_data=None, text_data_columns=None, number_data_columns=None, standardize=False):
        """
        load data
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

        self.text_data = text_data
        self.text_data_columns = text_data_columns

        if standardize:
            self.number_data = self._standardize(number_data)
        else:
            self.number_data = number_data
        self.number_data_columns = number_data_columns

    def fit_transform(self, metric=None, lens=None, scaler=preprocessing.MinMaxScaler()):
        """
        project data to point cloud
        """
        if self.number_data is None:
            raise ValueError("Data must not None.")

        self.metric = metric
        self.lens = lens
        self.scaler = scaler

        self.point_cloud = None

        d = self._create_dist_matrix(metric)
        d = self._project_data(d, lens)
        self.point_cloud = self._scale_data(d, scaler)

    def map(self, resolution=10, overlap=1, eps=1, min_samples=1):
        """
        map point cloud to topological space
        """
        if self.point_cloud is None:
            raise ValueError("Point cloud is not exist yet.")

        if self.point_cloud.ndim != 2:
            raise ValueError("Data must be 2d array.")

        if resolution <= 0:
            raise ValueError("Resolution must be greater than Zero.")

        if overlap < 0:
            raise ValueError("Overlap must be greater than Zero.")

        if eps < 0:
            raise ValueError("Eps must be greater than Zero.")

        if min_samples < 0:
            raise ValueError("Min samples must be greater than Zero.")

        self.resolution = resolution
        self.overlap = overlap
        self.eps = eps
        self.min_samples = min_samples

        self.hypercubes = {}

        dist_vec = self._calc_dist_vec(self.number_data)
        dist_eps = self._calc_dist_from_eps(dist_vec, eps)

        self._create_hypercubes(resolution, overlap, cluster.DBSCAN(eps=dist_eps, min_samples=min_samples))

        if self.verbose == 1:
            print("created %s nodes." % (len(self.hypercubes)))

        # nodeが0個なら終了
        if len(self.hypercubes) == 0:
            return

        self._calc_node_coordinate()
        self._calc_edges()

        if self.verbose == 1:
            if self.edges is None:
                print("created 0 edges.")
            else:
                print("created %s edges." % (len(self.edges)))

    def color(self, target, color_method="mean", color_type="rgb", normalize=False):
        """
        colored by target
        """
        if target is None:
            raise Exception("Target must not None.")

        if type(target) is not np.ndarray:
            target = np.array(target)

        if len(target) != self.number_data.shape[0]:
            raise ValueError("Target data size must be same with input data")

        if color_method not in ["mean", "mode"]:
            raise ValueError("Data type must be 'mean' or 'mode'.")

        if color_type not in ["rgb", "gray"]:
            raise ValueError("Color type must be 'rgb' or 'gray'.")

        if len(target.shape) == 1:
            target = target.reshape(-1, 1)

        self.color_target = target

        self.hex_colors = [""] * len(self.hypercubes)

        # targetを0~1に正規化する
        if normalize:
            scaled_target = self._normalize(target)
        else:
            scaled_target = target

        colors = self._calc_color_values(scaled_target, color_method)
        self.colors = self._rescale_colors(colors)
        self._calc_hex_color_values(color_type)

    def show(self, fig_size=(5, 5), node_size=5, edge_width=1, mode=None, strength=None):
        """
        show graph
        """
        if mode == "spring":
            self._spring_plot(fig_size, node_size, edge_width, strength)
        else:
            self._plot(fig_size, node_size, edge_width)
        plt.show()

    def save(self, filename, mode=None, fig_size=(5, 5), node_size=5, edge_width=1, strength=None):
        """
        save graph
        """
        if mode == "spring":
            self._spring_plot(fig_size, node_size, edge_width, strength)
        else:
            self._plot(fig_size, node_size, edge_width)
        plt.savefig(filename)

    def search_from_id(self, node_id):
        """
        search from id
        """
        data_ids = self.hypercubes[node_id]

        text_data = None
        if self.text_data is not None:
            text_data = self.text_data[data_ids]

        number_data = None
        if self.number_data is not None:
            number_data = self._re_standardize(self.number_data[data_ids])

        text_data_columns = None
        if self.text_data_columns is not None:
            text_data_columns = self.text_data_columns

        number_data_columns = None
        if self.number_data_columns is not None:
            number_data_columns = self.number_data_columns

        result = {
            "id": node_id,
            "coordinate": self.nodes[node_id],
            "data_ids": data_ids,
            "text_data": text_data,
            "text_data_columns": text_data_columns,
            "number_data": number_data,
            "number_data_columns": number_data_columns,
        }

        if self.verbose == 1:
            print("node id: {}".format(node_id))
            print("coordinate: {}".format(result["coordinate"]))
            print("data ids: {}".format(result["data_ids"]))
            print("text data columns:")
            print(result["text_data_columns"])
            print("text data:")
            print(result["text_data"])
            print("number data columns:")
            print(result["number_data_columns"])
            print("number data:")
            print(result["number_data"])
        return result

    def search_from_values(self, search_dicts=None, target=None, search_type="index"):
        """
        search from dict
        """
        if search_type not in ["column", "index"]:
            raise ValueError("Data type must be 'column' or 'index'.")

        if search_type == "column":
            if self.text_data is not None and self.text_data_columns is None:
                raise ValueError("When search type is column, text_data_columns must not None.")

            if self.number_data_columns is None:
                raise ValueError("When search type is column, number_data_columns must not None.")

        number_data = self._re_standardize(self.number_data)
        search_number_data, search_number_data_columns = self._concatenate_target(number_data, target)

        data_index = self._data_index_from_search_dict(search_number_data, search_number_data_columns, search_dicts, search_type)
        node_index = self._node_index_from_data_id(data_index)
        self._set_search_color(node_index)

        return node_index

    def output_csv_from_node_ids(self, filename, node_ids=[], target=None, skip_header=False):
        """
        output file
        """
        if len(node_ids) > 0:
            number_data = self._re_standardize(self.number_data)
            number_data, number_data_columns = self._concatenate_target(number_data, target)

            data_index = self._data_index_from_node_id(node_ids)
            if self.text_data is not None:
                extract_data = np.concatenate([self.text_data[data_index], number_data[data_index]], axis=1)
            else:
                extract_data = number_data[data_index]

            extract_data = pd.DataFrame(extract_data)

        if skip_header:
            extract_data.to_csv(filename, columns=None, index=None)
        else:
            columns = np.concatenate([self.text_data_columns, number_data_columns])
            extract_data.columns = columns
            extract_data.to_csv(filename, columns=extract_data.columns, index=None)


class Topology(TopologyCore):
    """
    TDA class
    """

    def __init__(self, verbose=1):
        super(Topology, self).__init__(verbose)
        self.point_cloud_hex_colors = None
        self.point_cloud_sizes = None

    def _calc_point_cloud_hex_color(self, target, color_type):
        for i, t in enumerate(target):
            if color_type == "rgb":
                hex_color = self._hex_color(t)
            elif color_type == "gray":
                hex_color = self._grayscale_color(t)

            self.point_cloud_hex_colors[i] = hex_color

    def _get_train_test_index(self, length, size=0.9):
        threshold = int(length * size)
        index = np.random.permutation(length)
        train_index = np.sort(index[:threshold])
        test_index = np.sort(index[threshold:])
        return train_index, test_index

    def _set_search_point_cloud_color(self, data_index):
        searched_color = ["#cccccc"] * len(self.point_cloud)
        for i in data_index:
            searched_color[i] = self.point_cloud_hex_colors[i]
        self.point_cloud_hex_colors = searched_color

    def color_point_cloud(self, target, color_type="rgb", normalize=False):
        """
        color point cloud by target
        """
        if target is None:
            raise Exception("Target must not None.")

        if type(target) is not np.ndarray:
            target = np.array(target)

        if len(target) != self.point_cloud.shape[0]:
            raise ValueError("Target data size must be same with point cloud data")

        if color_type not in ["rgb", "gray"]:
            raise ValueError("Color type must be 'rgb' or 'gray'.")

        if len(target.shape) == 1:
            target = target.reshape(-1, 1)

        self.point_cloud_hex_colors = [""] * len(self.point_cloud)

        # targetを0~1に正規化する
        if normalize:
            scaled_target = self._normalize(target)
        else:
            scaled_target = target

        scaled_target = self._rescale_colors(scaled_target)
        self._calc_point_cloud_hex_color(scaled_target, color_type)

    def supervised_clustering_point_cloud(self, clusterer=None, target=None, train_size=0.9):
        if clusterer is not None and "fit" in dir(clusterer) and target is not None:
            # 教師データとテストデータに分ける
            self.train_index, test_index = self._get_train_test_index(self.point_cloud.shape[0], train_size)
            x_train = self.point_cloud[self.train_index, :]
            x_test = self.point_cloud[test_index, :]
            y_train = target[self.train_index].astype(int)

            # 目的変数がラベルデータ(int)なら分類&predictする
            clusterer.fit(x_train, y_train)
            labels = np.zeros((self.point_cloud.shape[0], 1))
            labels[self.train_index] += y_train.reshape(-1, 1)
            labels[test_index] += clusterer.predict(x_test).reshape(-1, 1)
            labels = self._normalize(labels)
            self.point_cloud_hex_colors = [self._hex_color(i) for i in labels]

    def unsupervised_clustering_point_cloud(self, clusterer=None):
        if clusterer is not None and "fit" in dir(clusterer):
            clusterer.fit(self.point_cloud)
            labels = self._normalize(clusterer.labels_)
            self.point_cloud_hex_colors = [self._hex_color(i) if i >= 0 else "#000000" for i in labels]

    def show_point_cloud(self, fig_size=(5, 5), node_size=5):
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], c=self.point_cloud_hex_colors, s=node_size)
        plt.axis("off")
        plt.show()

    def search_point_cloud(self, search_dicts=None, target=None, search_type="index"):
        search_number_data = self._re_standardize(self.number_data)
        search_number_data, search_number_data_columns = self._concatenate_target(search_number_data, target)

        data_index = self._data_index_from_search_dict(search_number_data, search_number_data_columns, search_dicts, search_type)
        self._set_search_point_cloud_color(data_index)
        return data_index
