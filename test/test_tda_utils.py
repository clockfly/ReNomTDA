import pytest
import numpy as np
from numpy.testing import assert_array_equal
import sklearn.cluster as cluster
from renom_tda.utils import DistUtil, MapUtil, GraphUtil


def test_dist_util_matrix_cityblock():
    data = np.array([[0., 0.], [1., 1.]])

    m = DistUtil(metric="cityblock")
    dist_matrix = m.calc_dist_matrix(data)
    test_matrix = np.array([[0., 2.], [2., 0.]])

    assert_array_equal(dist_matrix, test_matrix)


def test_dist_util_matrix_euclidean():
    data = np.array([[0., 0.], [1., 1.]])

    m = DistUtil(metric="euclidean")
    dist_matrix = m.calc_dist_matrix(data)
    test_matrix = np.array([[0., np.sqrt(2)], [np.sqrt(2), 0.]])

    assert_array_equal(dist_matrix, test_matrix)


# def test_dist_util_matrix_cosine():
#     data = np.array([[2., 1.], [1., 2.]])

#     m = DistUtil(metric="cosine")
#     dist_matrix = m.calc_dist_matrix(data)

#     dist = 4 / (np.sqrt(5) * np.sqrt(5))
#     test_matrix = np.array([[0., 1 - dist], [1 - dist, 0.]])
#     assert_array_equal(dist_matrix, test_matrix)


def test_dist_util_matrix_hamming():
    data = np.array([[0., 0.], [0., 2.]])

    m = DistUtil(metric="hamming")
    dist_matrix = m.calc_dist_matrix(data)
    test_matrix = np.array([[0., 0.5], [0.5, 0.]])

    assert_array_equal(dist_matrix, test_matrix)


def test_dist_util_matrix_none_input():
    data = None
    m = DistUtil()
    with pytest.raises(Exception):
        m.calc_dist_matrix(data)


def test_dist_util_matrix_unusable_metric():
    with pytest.raises(Exception):
        DistUtil(metric="somemetric")


def test_dist_util_eps():
    data = np.array([[0., 0.], [1., 1.]])

    m = DistUtil(metric="cityblock")
    m.calc_dist_matrix(data)
    eps = m.calc_eps(0.5)
    test_eps = 1

    assert eps == test_eps


def test_map_util():
    clusterer = cluster.DBSCAN(eps=0.5, min_samples=3)
    u = MapUtil(resolution=2, overlap=0.3, clusterer=clusterer)

    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    hypercubes = u.map(data=data, point_cloud=data)
    print(hypercubes)

    test_hypercubes = {
        0: [0, 1, 2],
        1: [2, 6, 8],
        2: [2, 5, 7],
        3: [2, 3, 4]
    }

    assert hypercubes == test_hypercubes


def test_map_util_overlap():
    clusterer = cluster.DBSCAN(eps=1.5, min_samples=3)
    u = MapUtil(resolution=2, overlap=2, clusterer=clusterer)

    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    hypercubes = u.map(data=data, point_cloud=data)
    print(hypercubes)

    test_hypercubes = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8]}

    assert hypercubes == test_hypercubes


def test_map_util_none_data():
    clusterer = cluster.DBSCAN(eps=1, min_samples=3)
    u = MapUtil(resolution=2, overlap=0.3, clusterer=clusterer)

    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    with pytest.raises(Exception):
        u.map(data=None, point_cloud=data)


def test_map_util_none_point_cloud():
    clusterer = cluster.DBSCAN(eps=1, min_samples=3)
    u = MapUtil(resolution=2, overlap=0.3, clusterer=clusterer)

    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    with pytest.raises(Exception):
        u.map(data=data, point_cloud=None)


def test_map_util_none_clusterer():
    clusterer = None
    u = MapUtil(resolution=2, overlap=0.3, clusterer=clusterer)

    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    hypercubes = u.map(data=data, point_cloud=data)

    test_hypercubes = {
        0: [0, 1, 2],
        1: [2, 6, 8],
        2: [2, 5, 7],
        3: [2, 3, 4]
    }

    assert hypercubes == test_hypercubes


def test_map_util_resolution_zero():
    with pytest.raises(Exception):
        MapUtil(resolution=0, overlap=0.3, clusterer=None)


def test_map_util_overlap_zero():
    with pytest.raises(Exception):
        MapUtil(resolution=10, overlap=0, clusterer=None)


def test_graph_util_node():
    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    hypercubes = {
        0: [0, 1, 2],
        1: [2, 6, 8],
        2: [2, 5, 7],
        3: [2, 3, 4]
    }

    u = GraphUtil(point_cloud=data, hypercubes=hypercubes)
    nodes, node_sizes = u.calc_node_coordinate()

    test_nodes = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    test_sizes = np.array([[3], [3], [3], [3]])

    assert_array_equal(nodes, test_nodes)
    assert_array_equal(node_sizes, test_sizes)


def test_graph_util_edge():
    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    hypercubes = {
        0: [0, 1, 2],
        1: [2, 6, 8],
        2: [2, 5, 7],
        3: [2, 3, 4]
    }

    u = GraphUtil(point_cloud=data, hypercubes=hypercubes)
    edges = u.calc_edges()

    test_edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

    assert_array_equal(edges, test_edges)


def test_graph_util_color():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    hypercubes = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8]
    }

    target = np.array([[0], [0], [0],
                       [0.5], [0.5], [0.5],
                       [1], [1], [1]])

    u = GraphUtil(point_cloud=data, hypercubes=hypercubes)
    colors, hex_colors = u.color(target=target, color_method="mean", color_type="rgb")

    test_colors = np.array([[0], [0.5], [1]])
    test_hex_colors = ["#0000b2", "#00b200", "#b20000"]

    assert_array_equal(colors, test_colors)
    assert hex_colors == test_hex_colors


def test_graph_util_color_none_target():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    hypercubes = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8]
    }

    u = GraphUtil(point_cloud=data, hypercubes=hypercubes)
    with pytest.raises(Exception):
        u.color(target=None, color_method="mean", color_type="rgb")


def test_graph_util_color_unusable_method():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    hypercubes = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8]
    }

    target = np.array([[0], [0], [0],
                       [0.5], [0.5], [0.5],
                       [1], [1], [1]])

    u = GraphUtil(point_cloud=data, hypercubes=hypercubes)
    with pytest.raises(Exception):
        u.color(target=target, color_method="somemetod", color_type="rgb")


def test_graph_util_color_unusable_type():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    hypercubes = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8]
    }

    target = np.array([[0], [0], [0],
                       [0.5], [0.5], [0.5],
                       [1], [1], [1]])

    u = GraphUtil(point_cloud=data, hypercubes=hypercubes)
    with pytest.raises(Exception):
        u.color(target=target, color_method="mean", color_type="sometype")


def test_graph_util_none_point_cloud():
    hypercubes = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8]
    }

    with pytest.raises(Exception):
        GraphUtil(point_cloud=None, hypercubes=hypercubes)


def test_graph_util_none_hypercubes():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    with pytest.raises(Exception):
        GraphUtil(point_cloud=data, hypercubes=None)
