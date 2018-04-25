# -*- coding: utf-8 -*.t-
from __future__ import print_function
import itertools
import numpy as np
import scipy
from scipy.spatial import distance
from renom_tda.painter import PainterResolver


class DistUtil(object):
    """Class of distance utility."""

    def __init__(self, metric="euclidean"):
        self.metrics = ["braycurtis",
                        "canberra",
                        "chebyshev",
                        "cityblock",
                        "correlation",
                        "cosine",
                        "dice",
                        "euclidean",
                        "hamming",
                        "jaccard",
                        "kulsinski",
                        "mahalanobis",
                        "matching",
                        "minkowski",
                        "rogerstanimoto",
                        "russellrao",
                        "seuclidean",
                        "sokalmichener",
                        "sokalsneath",
                        "sqeuclidean",
                        "yule"]

        if metric not in self.metrics:
            raise Exception("Metric {} is not usable.".format(metric))
        self.metric = metric

    def calc_dist_matrix(self, data):
        """Function of creating distance matrix.

        Params:
            data：raw data.
        """
        if data is None:
            raise Exception("Data must not None.")

        self.dist_matrix = distance.cdist(np.array(data), np.array(data), metric=self.metric)
        return self.dist_matrix

    def calc_eps(self, eps):
        dist_min = np.min(self.dist_matrix)
        dist_max = np.max(self.dist_matrix)
        return dist_min + (dist_max - dist_min) * eps


class MapUtil(object):
    """Class of Mapping utility."""

    def __init__(self, resolution=10, overlap=1, clusterer=None):
        if resolution <= 0:
            raise Exception("Resolution must greater than 0.")

        if overlap <= 0:
            raise Exception("Overlap must greater than 0.")

        self.resolution = resolution
        self.overlap = overlap
        self.clusterer = clusterer

    def map(self, data, point_cloud):
        if data is None:
            raise Exception("Data must not None.")

        if point_cloud is None:
            raise Exception("Point cloud must not None.")

        hypercubes = {}

        # データのindex。hypercubeに含まれるデータを特定するのに使う。
        input_data_ids = np.arange(data.shape[0])

        # 作成したノードの数。hypercubeのidに使う。
        created_node_count = 0

        # 分割する間隔
        chunk_width = 1. / self.resolution
        # 重なり幅
        overlap_width = chunk_width * self.overlap

        # hypercubeに分割する
        # resolution: 10, point_cloudが2次元なら、10x10の100個のhypercubeを作る
        for i in itertools.product(np.arange(self.resolution), repeat=point_cloud.shape[1]):
            # hypercubeの上限、下限を設定する
            floor = np.array(i) * chunk_width - overlap_width
            roof = np.array([n + 1 for n in i]) * chunk_width + overlap_width

            # hypercubeの上限、下限からマスクを求める
            mask_floor = point_cloud > floor
            mask_roof = point_cloud < roof
            mask = np.all(mask_floor & mask_roof, axis=1)

            # hypercubeに含まれるidを求める。
            masked_data_ids = input_data_ids[mask]

            # hypercubeに含まれる元の次元のデータを求める。
            masked_data = data[mask]

            # hypercube内の点を元の次元の距離で再度クラスタリング
            if masked_data.shape[0] > 0:
                if (self.clusterer is not None) and ("fit" in dir(self.clusterer)):
                    self.clusterer.fit(masked_data)

                    # クラスタリング結果のラベルでループを回す
                    for label in np.unique(self.clusterer.labels_):
                        # ラベルが-1の点はノイズとする
                        if label != -1:
                            hypercube_value = list(masked_data_ids[self.clusterer.labels_ == label])
                            # 同じ値を持つhypercubeは作らない
                            if hypercube_value not in hypercubes.values():
                                hypercubes.update(
                                    {created_node_count: hypercube_value})
                                created_node_count += 1
                else:
                    hypercubes.update({created_node_count: list(masked_data_ids)})
                    created_node_count += 1

        return hypercubes


class GraphUtil(object):
    """Class of Graph utility."""

    def __init__(self, point_cloud, hypercubes):
        if point_cloud is None:
            raise Exception("point_cloud must not None.")

        if hypercubes is None:
            raise Exception("hypercubes must not None.")

        self.point_cloud = point_cloud
        self.hypercubes = hypercubes

    def calc_node_coordinate(self):
        # ノードの配列を(ノード数, 投影する次元数)の大きさで初期化する
        nodes = np.zeros((len(self.hypercubes), self.point_cloud.shape[1]))
        node_sizes = np.zeros((len(self.hypercubes), 1))

        # hypercubeのキーでループ
        for key in self.hypercubes.keys():
            # ノード内のデータを取得
            data_in_node = self.point_cloud[self.hypercubes[key]]
            # ノードの座標をノード内の点の座標の平均で求める
            node_coordinate = np.average(data_in_node, axis=0)
            # ノードの配列を更新する
            nodes[int(key)] += node_coordinate
            node_sizes[int(key)] += len(data_in_node)

        return nodes, node_sizes

    def calc_edges(self):
        # エッジの配列を初期化（最終的な大きさはわからないので[]で初期化）
        edges = []

        # hypercubeのキーの組み合わせでループ
        for keys in itertools.combinations(self.hypercubes.keys(), 2):
            cube1 = set(self.hypercubes[keys[0]])
            cube2 = set(self.hypercubes[keys[1]])

            # cube1とcube2で重なる点があったらエッジを作る
            if len(cube1.intersection(cube2)) > 0:
                edge = np.array(keys).reshape(1, -1)
                if len(edges) == 0:
                    edges = edge
                else:
                    edges = np.concatenate([edges, edge], axis=0)
        return edges

    def color(self, target, color_method, color_type):
        if target is None:
            raise Exception("Target must not None.")

        if color_method not in ["mean", "mode"]:
            raise Exception("color_method {} is not usable.".format(color_method))

        if color_type not in ["rgb", "gray"]:
            raise Exception("color_type {} is not usable.".format(color_type))

        painter_resolver = PainterResolver()
        painter = painter_resolver.resolve(color_type)

        colors = np.zeros((len(self.hypercubes), 1))
        hex_colors = [""] * len(self.hypercubes)

        for key in self.hypercubes.keys():
            target_in_node = target[self.hypercubes[key]]

            if color_method == "mean":
                v = np.mean(target_in_node)
            elif color_method == "mode":
                v = scipy.stats.mode(target_in_node)[0][0]

            colors[int(key)] = v
            hex_colors[int(key)] = painter.paint(v)

        return colors, hex_colors
