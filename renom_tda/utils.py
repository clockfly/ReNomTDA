# -*- coding: utf-8 -*.t-
from __future__ import print_function
import itertools
import numpy as np
import scipy
from scipy.spatial import distance
from renom_tda.painter import PainterResolver


class DistUtil(object):
    """Class of distance utility.

    Params:
        metric: distance metric.
    """

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
            data： raw data.

        Return:
            distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        self.dist_matrix = distance.cdist(np.array(data), np.array(data), metric=self.metric)
        return self.dist_matrix

    def calc_eps(self, eps):
        """Function of calculating eps from distance matrix.

        Params:
            eps: ratio from minimum value.

        Return:
            distance_eps: calculated distance.
        """
        dist_min = np.min(self.dist_matrix)
        dist_max = np.max(self.dist_matrix)
        return dist_min + (dist_max - dist_min) * eps


class MapUtil(object):
    """Class of Mapping utility.

    Params:
        resolution: The number of division of each axis.

        overlap: The width of overlapping of division.

        clusterer: Class of clustering method.
    """

    def __init__(self, resolution=10, overlap=1, clusterer=None):
        if resolution <= 0:
            raise Exception("Resolution must greater than 0.")

        if overlap <= 0:
            raise Exception("Overlap must greater than 0.")

        self.resolution = resolution
        self.overlap = overlap
        self.clusterer = clusterer

    def map(self, data, point_cloud):
        """Function of mapping data.

        Params:
            data: number data.

            point_cloud: point cloud data.

        Return:
            hypercubes: calculated hypercubes dictionaly.
        """
        if data is None:
            raise Exception("Data must not None.")

        if point_cloud is None:
            raise Exception("Point cloud must not None.")

        hypercubes = {}

        # original data index
        input_data_ids = np.arange(data.shape[0])

        # counts of created nodes.
        created_node_count = 0

        # chunked with each axis.
        chunk_width = 1. / self.resolution
        # overlap width.
        overlap_width = chunk_width * self.overlap

        # divide point cloud to subcluster.
        # If point_cloud is 2 dim and resolution is 10, created 100(10x10) subclusters.
        for i in itertools.product(np.arange(self.resolution), repeat=point_cloud.shape[1]):
            # calculate floor and roof of subcluster.
            floor = np.array(i) * chunk_width - overlap_width
            roof = np.array([n + 1 for n in i]) * chunk_width + overlap_width

            # make mask of subcluster data.
            mask_floor = point_cloud > floor
            mask_roof = point_cloud < roof
            mask = np.all(mask_floor & mask_roof, axis=1)

            # get id in subcluster.
            masked_data_ids = input_data_ids[mask]

            # get original data in subcluster.
            masked_data = data[mask]

            # clustering subcluster data.
            if masked_data.shape[0] > 0:
                if (self.clusterer is not None) and ("fit" in dir(self.clusterer)):
                    self.clusterer.fit(masked_data)

                    # loop with clustering labels.
                    for label in np.unique(self.clusterer.labels_):
                        # label -1 is noise.
                        if label != -1:
                            # data ids in hypercube.
                            hypercube_value = list(masked_data_ids[self.clusterer.labels_ == label])
                            # hypercube is not exists yet, update hypercubes dict.
                            if hypercube_value not in hypercubes.values():
                                hypercubes.update(
                                    {created_node_count: hypercube_value})
                                created_node_count += 1
                else:
                    # didn't clustering in subcluster.
                    hypercubes.update({created_node_count: list(masked_data_ids)})
                    created_node_count += 1

        return hypercubes


class GraphUtil(object):
    """Class of Graph utility.

    Params:
        point_cloud: point cloud data.

        hypercubes: hypercubes.
    """

    def __init__(self, point_cloud, hypercubes):
        if point_cloud is None:
            raise Exception("point_cloud must not None.")

        if hypercubes is None:
            raise Exception("hypercubes must not None.")

        self.point_cloud = point_cloud
        self.hypercubes = hypercubes

    def calc_node_coordinate(self):
        """Function of calculate node coordinate with point_cloud & hypercubes.

        Return:
            nodes: node coordinates.

            node_size: array of node sizes.
        """
        # initialize node coordinate.
        nodes = np.zeros((len(self.hypercubes), self.point_cloud.shape[1]))
        node_sizes = np.zeros((len(self.hypercubes), 1))

        # loop with hypercubes.
        for key in self.hypercubes.keys():
            # get point cloud data in node.
            data_in_node = self.point_cloud[self.hypercubes[key]]
            # calc node coordinate by average.
            node_coordinate = np.average(data_in_node, axis=0)
            # update node coordinate data.
            nodes[int(key)] += node_coordinate
            node_sizes[int(key)] += len(data_in_node)

        return nodes, node_sizes

    def calc_edges(self):
        """Function of calculate edge with hypercubes.

        Return:
            edges: edges of topology.
        """
        # initialize edges with empty array.
        edges = []

        # loop with conbination of hypercube's keys.
        for keys in itertools.combinations(self.hypercubes.keys(), 2):
            cube1 = set(self.hypercubes[keys[0]])
            cube2 = set(self.hypercubes[keys[1]])

            # If intersection of cubes exists, append edges.
            if len(cube1.intersection(cube2)) > 0:
                edge = np.array(keys).reshape(1, -1)
                if len(edges) == 0:
                    edges = edge
                else:
                    edges = np.concatenate([edges, edge], axis=0)
        return edges

    def color(self, target, color_method, color_type):
        """Function of calculate color of nodes with hypercubes.

        Params:
            target: Array of coloring value.

            color_method: Method of calculate color value. "mean" or "mode".

            color_type: "rgb" or "gray".

        Return:
            colors: array of node colors.

            hex_colors: array of node hex colors.
        """
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
