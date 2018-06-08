import pytest
import numpy as np
from numpy.testing import assert_array_equal
from sklearn import cluster, preprocessing
from renom_tda import PCA, L1Centrality, GaussianDensity, Topology


def test_load_data_none():
    data = None

    t = Topology()
    with pytest.raises(ValueError):
        t.load_data(data)


def test_load_data_array_data():
    data = [[0.0, 0.0], [1.0, 1.0]]

    t = Topology()
    t.load_data(data)

    assert_array_equal(t.number_data, np.array(data))


def test_load_data_ndarray_data():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])

    t = Topology()
    t.load_data(data)

    assert_array_equal(t.number_data, data)


def test_load_data_not_2d_array():
    data = [[[0.0, 0.0], [1.0, 1.0]]]

    t = Topology()
    with pytest.raises(ValueError):
        t.load_data(data)


def test_load_data_standardize_true():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])

    t = Topology()
    t.load_data(data, standardize=True)

    scaler = preprocessing.StandardScaler()
    test_data = scaler.fit_transform(data)
    assert_array_equal(t.std_number_data, test_data)


def test_load_data_number_data_columns():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    data_columns = ["data1", "data2"]

    t = Topology()
    t.load_data(data, number_data_columns=data_columns)

    assert_array_equal(t.number_data_columns, np.array(data_columns))


def test_load_data_number_data_columns_ndarray():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    data_columns = np.array(["data1", "data2"])

    t = Topology()
    t.load_data(data, number_data_columns=data_columns)

    assert_array_equal(t.number_data_columns, data_columns)


def test_load_data_text_data():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    text_data = [["data1"], ["data2"]]

    t = Topology()
    t.load_data(data, text_data=text_data)

    assert_array_equal(t.text_data, np.array(text_data))


def test_load_data_text_data_ndarray():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    text_data = np.array([["data1"], ["data2"]])

    t = Topology()
    t.load_data(data, text_data=text_data)

    assert_array_equal(t.text_data, text_data)


def test_load_data_text_data_columns():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    text_data = np.array([["data1-1", "data1-2"], ["data2-1", "data2-2"]])
    text_data_columns = ["columns1", "columns2"]

    t = Topology()
    t.load_data(data, text_data=text_data, text_data_columns=text_data_columns)

    assert_array_equal(t.text_data_columns, np.array(text_data_columns))


def test_load_data_text_data_columns_ndarray():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    text_data = np.array([["data1-1", "data1-2"], ["data2-1", "data2-2"]])
    text_data_columns = ["columns1", "columns2"]

    t = Topology()
    t.load_data(data, text_data=text_data, text_data_columns=text_data_columns)

    assert_array_equal(t.text_data_columns, text_data_columns)


def test_load_data_text_data_columns_text_data_is_none():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    text_data = None
    text_data_columns = ["columns1", "columns2"]

    t = Topology()
    with pytest.raises(ValueError):
        t.load_data(data, text_data=text_data, text_data_columns=text_data_columns)


def test_load_data_text_data_columns_text_data_diff_columns():
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    text_data = np.array([["data1-1", "data1-2"], ["data2-1", "data2-2"]])
    text_data_columns = ["columns1"]

    t = Topology()
    with pytest.raises(ValueError):
        t.load_data(data, text_data=text_data, text_data_columns=text_data_columns)


def test_transform_data_none():
    t = Topology()
    with pytest.raises(Exception):
        t.fit_transform()


def test_transform_none_none():
    data = np.array([[0., 0.], [1., 1.]])

    t = Topology()
    t.load_data(data)

    metric = None
    lens = None
    t.fit_transform(metric=metric, lens=lens)

    test_data = np.array([[0., 0.], [1., 1.]])

    assert_array_equal(t.point_cloud, test_data)


def test_transform_none_pca():
    data = np.array([[0., 1.], [1., 0.]])

    t = Topology()
    t.load_data(data)

    metric = None
    lens = [PCA(components=[0])]
    t.fit_transform(metric=metric, lens=lens)

    test_data = np.array([0., 1.])
    test_data = test_data.reshape(test_data.shape[0], 1)

    assert_array_equal(t.point_cloud, test_data)


def test_transform_multi_lens():
    data = np.array([[0., 0.], [0., 1.], [1., 1.]])

    t = Topology()
    t.load_data(data)

    metric = "hamming"
    lens = [L1Centrality(), GaussianDensity(h=0.25)]
    t.fit_transform(metric=metric, lens=lens)

    test_data = np.array([[1., 0.], [0., 1.], [1., 0.]])

    assert_array_equal(t.point_cloud, test_data)


def test_map():
    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    t = Topology()
    t.load_data(data)
    t.fit_transform(metric=None, lens=None)

    t.map(resolution=2, overlap=0.3, eps=0.3, min_samples=3)

    test_nodes = np.array([[0.25, 0.25],
                           [0.25, 0.75],
                           [0.75, 0.25],
                           [0.75, 0.75]])

    test_edges = np.array([[0, 1],
                           [0, 2],
                           [0, 3],
                           [1, 2],
                           [1, 3],
                           [2, 3]])

    assert_array_equal(t.nodes, test_nodes)
    assert_array_equal(t.edges, test_edges)


def test_map_point_cloud_none():
    t = Topology()
    with pytest.raises(Exception):
        t.map()


def test_map_resolution_under_zero():
    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    t = Topology()
    t.load_data(data)
    t.fit_transform(metric=None, lens=None)
    with pytest.raises(Exception):
        t.map(resolution=-1)


def test_map_overlap_under_zero():
    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    t = Topology()
    t.load_data(data)
    t.fit_transform(metric=None, lens=None)
    with pytest.raises(Exception):
        t.map(overlap=-1)


def test_map_eps_under_zero():
    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    t = Topology()
    t.load_data(data)
    t.fit_transform(metric=None, lens=None)
    with pytest.raises(Exception):
        t.map(eps=-1)


def test_map_min_samples_under_zero():
    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    t = Topology()
    t.load_data(data)
    t.fit_transform(metric=None, lens=None)
    with pytest.raises(Exception):
        t.map(min_samples=-1)


def test_color_mode_rgb():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    t = Topology()
    t.load_data(data)
    t.fit_transform(metric=None, lens=None)

    t.map(resolution=2, overlap=0.3, eps=0.2, min_samples=3)

    t.color(target, color_method="mode", color_type="rgb", normalize=True)

    test_color = ['#0000b2', '#00b200', '#b20000']

    assert t.hex_colors == test_color


def test_color_mean_rgb():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1.1], [0.9],
                       [2], [2], [2]])

    t = Topology()
    t.load_data(data)
    t.fit_transform(metric=None, lens=None)

    t.map(resolution=2, overlap=0.3, eps=0.2, min_samples=3)

    t.color(target, color_method="mean", color_type="rgb", normalize=True)

    test_color = ['#0000b2', '#00b200', '#b20000']

    assert t.hex_colors == test_color


def test_color_mode_gray():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    t = Topology()
    t.load_data(data)
    t.fit_transform(metric=None, lens=None)

    t.map(resolution=2, overlap=0.3, eps=0.2, min_samples=3)

    t.color(target, color_method="mode", color_type="gray", normalize=True)

    test_color = ['#dcdcdc', '#787878', '#141414']

    assert t.hex_colors == test_color


def test_color_mean_gray():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1.1], [0.9],
                       [2], [2], [2]])

    t = Topology()
    t.load_data(data)
    t.fit_transform(metric=None, lens=None)

    t.map(resolution=2, overlap=0.3, eps=0.2, min_samples=3)

    t.color(target, color_method="mean", color_type="gray", normalize=True)

    test_color = ['#dcdcdc', '#787878', '#141414']

    assert t.hex_colors == test_color


def test_color_none_input():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = None

    t = Topology()
    t.load_data(data)
    t.fit_transform()
    t.map()

    with pytest.raises(Exception):
        t.color(target)


def test_color_different_size_input():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([0, 1, 2])

    t = Topology()
    t.load_data(data)
    t.fit_transform()
    t.map()

    with pytest.raises(Exception):
        t.color(target)


def test_color_color_method():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    t = Topology()
    t.load_data(data)
    t.fit_transform()
    t.map(resolution=2, overlap=0.3)

    with pytest.raises(Exception):
        t.color(target, color_method="hoge")


def test_color_ctype():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    t = Topology()
    t.load_data(data)
    t.fit_transform()
    t.map(resolution=2, overlap=0.3)

    with pytest.raises(Exception):
        t.color(target, color_type="hoge")


def test_search_text_data():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    text_data = np.array([["a"], ["a"], ["a"],
                          ["b"], ["b"], ["b"],
                          ["c"], ["c"], ["c"]])

    t = Topology()
    t.load_data(data, text_data=text_data)
    t.fit_transform(metric=None, lens=None)
    t.map(resolution=2, overlap=0.3)
    t.color(target, color_method="mean", color_type="rgb", normalize=True)
    search_dicts = [{
        "data_type": "text",
        "operator": "=",
        "column": 0,
        "value": "a"
    }]
    t.search_from_values(search_dicts=search_dicts, target=None)

    test_color = ['#0000b2', '#cccccc', '#cccccc']
    assert t.hex_colors == test_color


def test_search_number_data():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    text_data = np.array([["a"], ["a"], ["a"],
                          ["b"], ["b"], ["b"],
                          ["c"], ["c"], ["c"]])

    t = Topology()
    t.load_data(data, text_data=text_data)
    t.fit_transform(metric=None, lens=None)
    t.map(resolution=2, overlap=0.3)
    t.color(target, color_method="mean", color_type="rgb", normalize=True)
    search_dicts = [{
        "data_type": "number",
        "operator": ">",
        "column": 0,
        "value": 0.7
    }]
    t.search_from_values(search_dicts=search_dicts, target=None, search_type="and")

    test_color = ['#cccccc', '#cccccc', '#b20000']
    assert t.hex_colors == test_color


def test_search_multiple_values():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    text_data = np.array([["a"], ["a"], ["a"],
                          ["b"], ["b"], ["b"],
                          ["c"], ["c"], ["c"]])

    t = Topology()
    t.load_data(data, text_data=text_data)
    t.fit_transform(metric=None, lens=None)
    t.map(resolution=2, overlap=0.3)
    t.color(target, color_method="mean", color_type="rgb", normalize=True)
    search_dicts = [{
        "data_type": "number",
        "operator": "<",
        "column": 0,
        "value": 0.3
    }, {
        "data_type": "text",
        "operator": "like",
        "column": 0,
        "value": "a"
    }]
    t.search_from_values(search_dicts=search_dicts, target=None, search_type="and")

    test_color = ['#0000b2', '#cccccc', '#cccccc']
    assert t.hex_colors == test_color


def test_search_target_data():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    columns = np.array(["columns1", "columns2"])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    text_data = np.array([["a"], ["a"], ["a"],
                          ["b"], ["b"], ["b"],
                          ["c"], ["c"], ["c"]])
    text_data_columns = np.array(["text_columns"])

    t = Topology()
    t.load_data(data, number_data_columns=columns, text_data=text_data, text_data_columns=text_data_columns)
    t.fit_transform(metric=None, lens=None)
    t.map(resolution=2, overlap=0.3)
    t.color(target, color_method="mean", color_type="rgb", normalize=True)
    search_dicts = [{
        "data_type": "number",
        "operator": "=",
        "column": -1,
        "value": 2
    }]
    t.search_from_values(search_dicts=search_dicts, target=target, search_type="and")

    test_color = ['#cccccc', '#cccccc', '#b20000']
    assert t.hex_colors == test_color
