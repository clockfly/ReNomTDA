# coding: utf-8

import colorsys

import _pickle as cPickle

import datetime

import json

import os

import numpy as np

import pandas as pd

import pkg_resources

import random

import scipy

import sqlite3

import string

from bottle import HTTPResponse, request, response, route, run, static_file

from sklearn import cluster, ensemble, neighbors, preprocessing, svm

import renom as rm
from renom.optimizer import Adam

from renom_tda.lens import PCA, TSNE, MDS
from renom_tda.lens_renom import AutoEncoder
from renom_tda.topology import SearchableTopology

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STRAGE_DIR = os.path.join(BASE_DIR, ".storage")


class Storage:
    def __init__(self):
        if not os.path.isdir(STRAGE_DIR):
            os.makedirs(STRAGE_DIR)

        dbname = os.path.join(STRAGE_DIR, 'storage.db')
        self.db = sqlite3.connect(dbname,
                                  check_same_thread=False,
                                  detect_types=sqlite3.PARSE_DECLTYPES,
                                  isolation_level=None)

        self.db.execute('PRAGMA journal_mode = WAL')
        self.db.execute('PRAGMA synchronous = OFF')
        self._init_db()

    def cursor(self):
        return self.db.cursor()

    def _init_db(self):
        c = self.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS topology_data
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         rand_str TEXT,
                         column_number INTEGER,
                         algorithm INTEGER,
                         analysis_mode INTEGER,
                         resolution INTEGER,
                         overlap NUMBER,
                         eps NUMBER,
                         min_samples INTEGER,
                         color_index INTEGER,
                         input_data BLOB,
                         categorical_data BLOB,
                         point_cloud BLOB,
                         nodes BLOB,
                         node_sizes BLOB,
                         edges BLOB,
                         colors BLOB,
                         hypercubes BLOB,
                         filename TEXT,
                         pca_result BLOB,
                         created TIMESTAMP,
                         updated TIMESTAMP,
                         UNIQUE(rand_str, column_number))
                  ''')

    def regist_topology_data(self, rand_str, column_number):
        with self.db:
            c = self.cursor()
            input_data = cPickle.dumps([])
            categorical_data = cPickle.dumps([])
            point_cloud = cPickle.dumps([])
            nodes = cPickle.dumps([])
            edges = cPickle.dumps([])
            colors = cPickle.dumps([])
            node_sizes = cPickle.dumps([])
            hypercubes = cPickle.dumps([])
            pca = cPickle.dumps([])
            now = datetime.datetime.now()
            c.execute('''
                INSERT INTO
                    topology_data(rand_str, column_number, input_data, categorical_data,
                    point_cloud, nodes, edges, colors, node_sizes, hypercubes, pca_result, created)
                VALUES
                    (?, ?, ?, ?, ?, ? , ?, ?, ?, ?, ?, ?)
                ''', (rand_str, column_number, input_data, categorical_data,
                      point_cloud, nodes, edges, colors, node_sizes, hypercubes, pca, now))

    def update_topology_data(self, rand_str, column_number, canvas_params, topology, categorical_data,
                             nodes, edges, colors, sizes, filename, pca_result):
        with self.db:
            c = self.cursor()
            input_data = cPickle.dumps(topology.input_data)
            categorical_data = cPickle.dumps(categorical_data)
            point_cloud = cPickle.dumps(topology.point_cloud)
            nodes = cPickle.dumps(nodes)
            edges = cPickle.dumps(edges)
            colors = cPickle.dumps(colors)
            sizes = cPickle.dumps(sizes)
            hypercubes = cPickle.dumps(topology.hypercubes)
            pca = cPickle.dumps(pca_result)
            now = datetime.datetime.now()
            c.execute('''
                UPDATE
                    topology_data
                SET
                    algorithm=?, analysis_mode=?, resolution=?, overlap=?, eps=?, min_samples=?,
                    color_index=?, input_data=?, categorical_data=?, point_cloud=?, nodes=?,
                    node_sizes=?, edges=?, colors=?, hypercubes=?, filename=?, pca_result=?, updated=?
                WHERE
                    rand_str=?
                    AND
                    column_number=?
                ''', (canvas_params[column_number]["algorithm"],
                      canvas_params[column_number]["mode"],
                      canvas_params[column_number]["resolution"],
                      canvas_params[column_number]["overlap"],
                      canvas_params[column_number]["eps"],
                      canvas_params[column_number]["min_samples"],
                      canvas_params[column_number]["color_index"],
                      input_data, categorical_data, point_cloud,
                      nodes, sizes, edges, colors, hypercubes, filename, pca,
                      now, rand_str, column_number
                      ))

    def get_topology_data(self, rand_str):
        with self.db:
            c = self.cursor()
            c.execute('''
                SELECT
                    column_number, algorithm, analysis_mode, resolution, overlap, eps,
                    min_samples, color_index, input_data, categorical_data, point_cloud,
                    nodes, node_sizes, edges, colors, hypercubes, filename, pca_result
                FROM
                    topology_data
                WHERE
                    rand_str=?
                ''', (rand_str,))

            topology_data = {}
            for data in c.fetchall():
                topology_data.update({
                    data[0]: {
                        "algorithm": data[1],
                        "mode": data[2],
                        "resolution": data[3],
                        "overlap": data[4],
                        "eps": data[5],
                        "min_samples": data[6],
                        "color_index": data[7],
                        "input_data": cPickle.loads(data[8], encoding='ascii'),
                        "categorical_data": cPickle.loads(data[9], encoding='ascii'),
                        "point_cloud": cPickle.loads(data[10], encoding='ascii'),
                        "nodes": cPickle.loads(data[11], encoding='ascii'),
                        "sizes": cPickle.loads(data[12], encoding='ascii'),
                        "edges": cPickle.loads(data[13], encoding='ascii'),
                        "colors": cPickle.loads(data[14], encoding='ascii'),
                        "hypercubes": cPickle.loads(data[15], encoding='ascii'),
                        "filename": data[16],
                        "pca_result": cPickle.loads(data[17], encoding='ascii'),
                    }
                })
        return topology_data


class AutoEncoder2Layer(rm.Model):
    def __init__(self, unit_size):
        self._encodelayer = rm.Dense(10)
        self._encodedlayer = rm.Dense(2)
        self._decodelayer = rm.Dense(10)
        self._decodedlayer = rm.Dense(unit_size)

    def forward(self, x):
        el_out = rm.relu(self._encodelayer(x))
        l = rm.relu(self._encodedlayer(el_out))
        dl_out = rm.relu(self._decodelayer(l))
        g = self._decodedlayer(dl_out)
        loss = rm.mse(g, x)
        return loss

    def encode(self, x):
        el_out = rm.relu(self._encodelayer(x))
        l = self._encodedlayer(el_out)
        return l


class AutoEncoder3Layer(rm.Model):
    def __init__(self, unit_size):
        self._encodelayer1 = rm.Dense(25)
        self._encodelayer2 = rm.Dense(10)
        self._encodedlayer = rm.Dense(2)
        self._decodelayer1 = rm.Dense(10)
        self._decodelayer2 = rm.Dense(25)
        self._decodedlayer = rm.Dense(unit_size)

    def forward(self, x):
        el1_out = rm.relu(self._encodelayer1(x))
        el2_out = rm.relu(self._encodelayer2(el1_out))
        l = rm.relu(self._encodedlayer(el2_out))
        dl1_out = rm.relu(self._decodelayer1(l))
        dl2_out = rm.relu(self._decodelayer2(dl1_out))
        g = self._decodedlayer(dl2_out)
        loss = rm.mse(g, x)
        return loss

    def encode(self, x):
        el1_out = rm.relu(self._encodelayer1(x))
        el2_out = rm.relu(self._encodelayer2(el1_out))
        l = self._encodedlayer(el2_out)
        return l


class AutoEncoder4Layer(rm.Model):
    def __init__(self, unit_size):
        self._encodelayer1 = rm.Dense(50)
        self._encodelayer2 = rm.Dense(25)
        self._encodelayer3 = rm.Dense(10)
        self._encodedlayer = rm.Dense(2)
        self._decodelayer1 = rm.Dense(10)
        self._decodelayer2 = rm.Dense(25)
        self._decodelayer3 = rm.Dense(50)
        self._decodedlayer = rm.Dense(unit_size)

    def forward(self, x):
        el1_out = rm.relu(self._encodelayer1(x))
        el2_out = rm.relu(self._encodelayer2(el1_out))
        el3_out = rm.relu(self._encodelayer3(el2_out))
        l = rm.relu(self._encodedlayer(el3_out))
        dl1_out = rm.relu(self._decodelayer1(l))
        dl2_out = rm.relu(self._decodelayer2(dl1_out))
        dl3_out = rm.relu(self._decodelayer3(dl2_out))
        g = self._decodedlayer(dl3_out)
        loss = rm.mse(g, x)
        return loss

    def encode(self, x):
        el1_out = rm.relu(self._encodelayer1(x))
        el2_out = rm.relu(self._encodelayer2(el1_out))
        el3_out = rm.relu(self._encodelayer3(el2_out))
        l = self._encodedlayer(el3_out)
        return l


def set_json_body(body):
    r = HTTPResponse(status=200, body=body)
    r.set_header('Content-Type', 'application/json')
    return r


def _get_statistic_value(data):
    q1 = scipy.stats.scoreatpercentile(data, 25)
    median = np.median(data)
    q3 = scipy.stats.scoreatpercentile(data, 75)
    return q1, median, q3


def _rescale_target(target):
    ret_target = np.zeros((len(target), 1))

    # 統計量を求める
    q1, median, q3 = _get_statistic_value(target)

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
    return ret_target


def _get_hex_color(i):
    # hsv色空間でiが0~1を青~赤に対応させる。
    c = colorsys.hsv_to_rgb((1 - i) * 240 / 360, 1.0, 0.7)
    return "#%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


@route("/")
def index():
    rand_str = request.get_cookie("rand_str")
    if rand_str is None:
        cookie_str = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(20)])
        response.set_cookie(
            "rand_str",
            cookie_str,
            max_age=315360000)

        storage.regist_topology_data(cookie_str, 0)
        storage.regist_topology_data(cookie_str, 1)
    return pkg_resources.resource_string(__name__, "index.html")


@route("/css/<file_name>")
def css(file_name):
    return static_file(file_name, root=BASE_DIR + '/css')


@route("/static/<file_name>")
def static(file_name):
    return pkg_resources.resource_string(__name__, "static/" + file_name)


@route("/fonts/<file_name>")
def fonts(file_name):
    return pkg_resources.resource_string(__name__, "static/fonts/" + file_name)


@route("/api/reset", method="GET")
def reset():
    return


def _get_data_and_index(filepath):
    # nanを含むデータは使わない
    pdata = pd.read_csv(filepath)
    # 文字列データと数値データを分ける
    categorical_data_index = (pdata.dtypes == "object")
    numerical_data_index = np.logical_or(pdata.dtypes == "float", pdata.dtypes == "int")
    return pdata, categorical_data_index, numerical_data_index


# fileのロード
# TODO
# ディレクトリ決め打ちを変更
@route("/api/load_file", method="POST")
def load_file():
    filename = request.params.filename
    filepath = os.path.join(DATA_DIR, filename)

    try:
        pdata, categorical_data_index, numerical_data_index = _get_data_and_index(filepath)

        if pdata.shape[0] > 10000:
            body = json.dumps({"error": "LF0002"})
            r = set_json_body(body)
            return r

        # 表示するカラム名や統計情報を取得する
        labels = pdata.columns
        categorical_data_labels = labels[categorical_data_index]
        # 数値データのラベル
        numerical_data_labels = labels[numerical_data_index]

        # 数値データはトポロジーの作成に使う項目と色をつける項目の絞り込みで使う
        numerical_data = np.array(pdata.loc[:, numerical_data_index])

        # 統計情報を取得する
        numerical_data_mins = numerical_data.min(axis=0)
        numerical_data_mins = np.around(numerical_data_mins, 2)
        numerical_data_maxs = numerical_data.max(axis=0)
        numerical_data_maxs = np.around(numerical_data_maxs, 2)
        numerical_data_means = numerical_data.mean(axis=0)
        numerical_data_means = np.around(numerical_data_means, 2)

        body = json.dumps({"numerical_data": numerical_data.tolist(),
                           "categorical_data_labels": categorical_data_labels.tolist(),
                           "numerical_data_labels": numerical_data_labels.tolist(),
                           "numerical_data_mins": numerical_data_mins.tolist(),
                           "numerical_data_maxs": numerical_data_maxs.tolist(),
                           "numerical_data_means": numerical_data_means.tolist()})
        r = set_json_body(body)
        return r

    except IOError:
        body = json.dumps({"error": "LF0001"})
        r = set_json_body(body)
        return r


# ノードをクリックした時に呼び出す関数
@route("/api/click", method="POST")
def click():
    rand_str = request.get_cookie("rand_str")

    filename = request.params.filename
    filepath = os.path.join(DATA_DIR, filename)
    node_index = int(request.params.clicknode)
    columns = int(request.params.columns)
    mode = int(request.params.mode)

    db_data = storage.get_topology_data(rand_str)

    topology = SearchableTopology(verbose=0)
    _set_canvas_data(topology, db_data[columns])

    # データ読み込み
    pdata, categorical_data_index, numerical_data_index = _get_data_and_index(filepath)
    numerical_data = np.array(pdata.loc[:, numerical_data_index])
    categorical_data = np.array(pdata.loc[:, categorical_data_index])

    if mode == 0 or mode == 1:
        numerical_data = np.around(numerical_data[node_index], 2)
        body = json.dumps({"categorical_data": [categorical_data[node_index, :].tolist()],
                           "data": [numerical_data.tolist()]})
    elif mode == 2:
        categorical_data = categorical_data[topology.hypercubes[node_index]]
        numerical_data = numerical_data[topology.hypercubes[node_index]]
        # 少数の桁を丸める
        numerical_data = np.around(numerical_data, 2)

        body = json.dumps({"categorical_data": categorical_data.tolist(), "data": numerical_data.tolist()})

    r = set_json_body(body)
    return r


def _dimension_reduction(topology, params, calc_data):
    # 平均、分散は標準化済みのデータをもとに戻すのに使う
    calc_data_avg = np.average(calc_data, axis=0)
    calc_data_std = np.std(calc_data, axis=0)
    normalized_calc_data = (calc_data - calc_data_avg) / (calc_data_std + 1e-6)
    algorithms = [PCA(components=[0, 1]),
                  TSNE(components=[0, 1]),
                  MDS(components=[0, 1]),
                  AutoEncoder(epoch=200,
                              batch_size=100,
                              network=AutoEncoder2Layer(normalized_calc_data.shape[1]),
                              opt=Adam()),
                  AutoEncoder(epoch=200,
                              batch_size=100,
                              network=AutoEncoder3Layer(normalized_calc_data.shape[1]),
                              opt=Adam()),
                  AutoEncoder(epoch=200,
                              batch_size=100,
                              network=AutoEncoder4Layer(normalized_calc_data.shape[1]),
                              opt=Adam()),
                  None]
    # 表示が切れるので、0~1ではなく0.01~0.99に正規化
    scaler = preprocessing.MinMaxScaler(feature_range=(0.01, 0.99))
    topology.fit_transform(normalized_calc_data, metric=None, lens=[
                           algorithms[params["algorithm"]]], scaler=scaler)


def _get_point_cloud_color(cdata):
    colorlist = _rescale_target(cdata)
    return [_get_hex_color(i) for i in colorlist]


def _get_train_test_index(length, size=0.9):
    threshold = int(length * size)
    index = np.random.permutation(length)
    train_index = np.sort(index[:threshold])
    test_index = np.sort(index[threshold:])
    return train_index, test_index


def _normalize(data):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(data)


def _get_clustering_color(topo, cdata, params, dist_vec):
    eps = dist_vec.min() + (dist_vec.max() - dist_vec.min()) * params["eps"]
    if params["mode"] == 1:
        clusterers = [cluster.KMeans(n_clusters=params["class_count"]),
                      cluster.DBSCAN(eps=eps, min_samples=params["min_samples"])]
    elif params["mode"] == 2:
        clusterers = [neighbors.KNeighborsClassifier(n_neighbors=params["neighbors"]),
                      svm.SVC(),
                      ensemble.RandomForestClassifier()]
    clusterer = clusterers[params["clustering_index"]]
    train_index = np.array([])

    # K-meansかDBSCANの時、教師なし分類
    if clusterer.__class__.__name__ == "KMeans" or clusterer.__class__.__name__ == "DBSCAN":
        clusterer.fit(topo.point_cloud)
        labels = _normalize(clusterer.labels_)
        label_colors = [_get_hex_color(i) if i >= 0 else "#000000" for i in labels]
    # KNN, SVM, RandomForestのとき教師あり分類
    else:
        # 教師データとテストデータに分ける
        train_index, test_index = _get_train_test_index(topo.point_cloud.shape[0], params["train_size"])
        x_train = topo.point_cloud[train_index, :]
        x_test = topo.point_cloud[test_index, :]
        y_train = cdata[train_index].astype(int)
        # y_test = cdata[test_index]
        # 目的変数がラベルデータ(int)なら分類&predictする
        clusterer.fit(x_train, y_train.reshape(-1,))
        labels = np.zeros((topo.point_cloud.shape[0], 1))
        labels[train_index] += y_train
        labels[test_index] += clusterer.predict(x_test).reshape(-1, 1)
        labels = _normalize(labels)
        label_colors = [_get_hex_color(i) for i in labels]

    return label_colors, train_index.tolist()


def _get_topology_color(topo, cdata):
    # トポロジーの色の設定
    topo.color(cdata)
    return topo.colors


def _get_topology_sizes(topo, resolution):
    # トポロジーの大きさの調整
    max_scale = 80 / resolution
    if max_scale > 2:
        max_scale = 2.0
    scaler = preprocessing.MinMaxScaler(feature_range=(0.3, max_scale))
    topo.node_sizes = scaler.fit_transform(topo.node_sizes)
    return topo.node_sizes.tolist()


def _check_file_and_algorithm(db_data, params, filename):
    if db_data["filename"] != filename:
        return True
    if db_data["algorithm"] != params["algorithm"]:
        return True
    return False


def _check_analysis_mode(db_data, params):
    if db_data["mode"] != params["mode"]:
        return True
    return False


def _check_params_change(db_data, params, filename):
    if db_data["filename"] != filename:
        return True
    if db_data["resolution"] != params["resolution"]:
        return True
    if db_data["overlap"] != params["overlap"]:
        return True
    if db_data["eps"] != params["eps"]:
        return True
    if db_data["min_samples"] != params["min_samples"]:
        return True
    return False


def _calc_dist_vec(topology):
    # 距離のベクトル(小さい順)を返す
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(topology.input_data)
    dist_mat = scipy.spatial.distance.cdist(scaled_data, scaled_data)
    dist_vec = np.trim_zeros(np.unique(dist_mat))
    return dist_vec


def _get_canvas_data_from_db_data(db_data):
    nodes = db_data["nodes"]
    edges = db_data["edges"]
    sizes = db_data["sizes"]
    return nodes, edges, sizes


def _set_topology_data(topology, db_data):
    topology.input_data = db_data["input_data"]
    topology.point_cloud = db_data["point_cloud"]
    topology.hypercubes = db_data["hypercubes"]


def _create(rand_str, canvas_params, calc_data, color_data, categorical_data, db_data, filename):
    canvas_data = {}
    pca_result = {
        "axis": None,
        "contribution_ratio": 0,
        "top_index": None
    }

    for key in canvas_params.keys():
        # インスタンス初期化
        topology = SearchableTopology(verbose=0)
        train_index = []

        if _check_file_and_algorithm(db_data[key], canvas_params[key], filename):
            # アルゴリズムが変わっていたら次元削減
            _dimension_reduction(topology, canvas_params[key], calc_data)
            if canvas_params[key]["algorithm"] == 0:
                sort_index = np.argsort(topology.lens[0].axis, axis=1)
                pca_result["axis"] = np.around(topology.lens[0].axis, 3).tolist()
                pca_result["contribution_ratio"] = np.around(np.sum(topology.lens[0].contribution_ratio), 3)
                pca_result["top_index"] = sort_index[:, -3:].tolist()
        else:
            # アルゴリズムが変わっていなければDBの値を使う
            _set_topology_data(topology, db_data[key])
            pca_result = db_data[key]["pca_result"]

        cdata = color_data[:, canvas_params[key]["color_index"]].reshape(-1, 1)

        # scatter plot
        if canvas_params[key]["mode"] == 0:
            nodes = topology.point_cloud.tolist()
            edges = []
            sizes = [0.3] * len(topology.point_cloud)
            colors = _get_point_cloud_color(cdata)
        # clustering
        elif canvas_params[key]["mode"] == 1:
            dist_vec = _calc_dist_vec(topology)
            nodes = topology.point_cloud.tolist()
            edges = []
            sizes = [0.3] * len(topology.point_cloud)
            colors, train_index = _get_clustering_color(topology, cdata, canvas_params[key], dist_vec)
        elif canvas_params[key]["mode"] == 2:
            dist_vec = _calc_dist_vec(topology)
            nodes = topology.point_cloud.tolist()
            edges = []
            sizes = [0.3] * len(topology.point_cloud)
            colors, train_index = _get_clustering_color(topology, cdata, canvas_params[key], dist_vec)
        # TDA
        elif canvas_params[key]["mode"] == 3:
            # パラメータが変わっていたらトポロジーの再計算
            check_file_and_algorithm = _check_file_and_algorithm(db_data[key], canvas_params[key], filename)
            check_analysis_mode = _check_analysis_mode(db_data[key], canvas_params[key])
            check_params = _check_params_change(db_data[key], canvas_params[key], filename)
            if check_file_and_algorithm or check_analysis_mode or check_params:
                dist_vec = _calc_dist_vec(topology)
                eps = dist_vec.min() + (dist_vec.max() - dist_vec.min()) * canvas_params[key]["eps"]
                topology.map(resolution=canvas_params[key]["resolution"],
                             overlap=canvas_params[key]["overlap"],
                             clusterer=cluster.DBSCAN(eps=eps,
                                                      min_samples=canvas_params[key]["min_samples"]))
                # nodeが作れなかったときreturnする
                if topology.nodes is None or len(topology.nodes) < 5:
                    canvas_data.update({key: {"nodes": [], "edges": [], "colors": [], "sizes": []}})
                    return canvas_data

                nodes = topology.nodes.tolist()
                edges = topology.edges.tolist()
                sizes = _get_topology_sizes(topology, canvas_params[key]["resolution"])
            # パラメータが変わっていなければdbのデータを使う
            else:
                nodes, edges, sizes = _get_canvas_data_from_db_data(db_data[key])
            colors = _get_topology_color(topology, cdata)

        storage.update_topology_data(rand_str, key, canvas_params, topology, categorical_data,
                                     nodes, edges, colors, sizes, filename, pca_result)
        canvas_data.update({key: {
            "nodes": nodes,
            "edges": edges,
            "colors": colors,
            "sizes": sizes,
            "train_index": train_index
        }})

    return canvas_data, pca_result


def _set_histogram_data(color_data, canvas_params):
    histogram_data = {}
    for key in canvas_params.keys():
        cdata = color_data[:, canvas_params[key]["color_index"]].reshape(-1, 1)
        data_color_values = cdata.reshape(-1,).tolist()
        q1, median, q3 = _get_statistic_value(cdata)
        statistic_value = [q1, median, q3]
        histogram_data.update({key: {
            "data_color_values": data_color_values,
            "statistic_value": statistic_value
        }})
    return histogram_data


# トポロジを作成する関数
@route("/api/create", method="POST")
def create():
    rand_str = request.get_cookie("rand_str")

    # パラメータ取得
    filename = request.params.filename
    filepath = os.path.join(DATA_DIR, filename)
    # トポロジーを計算する項目と色をつける項目のインデックスのリストを受け取る
    create_topology_index = list(map(int, request.params.create_topology_index.split(",")))
    colorize_topology_index = list(map(int, request.params.colorize_topology_index.split(",")))

    # canvas params
    # モード 0:scatter plot, 1:Clustering, 2:TDA
    canvas_params = {
        0: {
            "mode": int(request.params.mode),
            "color_index": int(request.params.color_index),
            "algorithm": int(request.params.algorithm),
            "resolution": int(request.params.resolution),
            "overlap": float(request.params.overlap),
            "clustering_index": int(request.params.clustering_index),
            "class_count": int(request.params.class_count),
            "eps": float(request.params.eps),
            "min_samples": float(request.params.min_samples),
            "train_size": float(request.params.train_size),
            "neighbors": int(request.params.neighbors),
        },
        1: {
            "mode": int(request.params.mode_col2),
            "color_index": int(request.params.color_index_col2),
            "algorithm": int(request.params.algorithm_col2),
            "resolution": int(request.params.resolution_col2),
            "overlap": float(request.params.overlap_col2),
            "clustering_index": int(request.params.clustering_index_col2),
            "class_count": int(request.params.class_count_col2),
            "eps": float(request.params.eps_col2),
            "min_samples": float(request.params.min_samples_col2),
            "train_size": float(request.params.train_size_col2),
            "neighbors": int(request.params.neighbors_col2),
        }
    }

    db_data = storage.get_topology_data(rand_str)

    # データ読み込み
    pdata, categorical_data_index, numerical_data_index = _get_data_and_index(filepath)
    numerical_data = np.array(pdata.loc[:, numerical_data_index])
    # 色付けに使うデータを抽出
    color_data = numerical_data[:, colorize_topology_index]
    histogram_data = _set_histogram_data(color_data, canvas_params)
    # トポロジーの作成に使うデータを抽出
    calc_data = numerical_data[:, create_topology_index]
    # カテゴリデータを抽出
    categorical_data = np.array(pdata.loc[:, categorical_data_index])

    # トポロジーを作る
    canvas_data, pca_result = _create(rand_str, canvas_params, calc_data,
                                      color_data, categorical_data, db_data, filename)

    # nodeが作れなかったときにエラーを返す
    if len(canvas_data[0]["nodes"]) < 5 or len(canvas_data[1]["nodes"]) < 5:
        body = json.dumps({"error": True})
    else:
        # bodyを設定
        body = json.dumps({"canvas_data": canvas_data,
                           "histogram_data": histogram_data,
                           "pca_result": pca_result
                           })
    r = set_json_body(body)
    return r


def _decode_txt(txt):
    # postデータのバイナリから元の文字列を復元
    # TODO
    bytes_str = b""
    for i in txt.split(","):
        bytes_str += int(i).to_bytes(1, byteorder='big')
    decoded_txt = bytes_str.decode('utf-8')
    return decoded_txt


def _set_canvas_data(topology, db_data):
    topology.input_data = db_data["input_data"]
    topology.categorical_data = db_data["categorical_data"]
    topology.point_cloud = db_data["point_cloud"]
    topology.hypercubes = db_data["hypercubes"]
    topology.nodes = db_data["nodes"]
    topology.edges = db_data["edges"]
    topology.colors = db_data["colors"]
    topology.sizes = db_data["sizes"]


def _search_node(topo, value, colorlist, is_categorical, operator, column_index, cdata):
    searched_color = ["#cccccc"] * len(topo.categorical_data)
    index = []

    if is_categorical == 0:
        if column_index < topo.input_data.shape[1]:
            search_data = topo.input_data[:, column_index]
        else:
            search_data = (cdata - cdata.mean()) / (cdata.std() + 1e-6)
        if operator == "=":
            index.extend(np.where(search_data == value)[0])
        elif operator == ">":
            index.extend(np.where(search_data > value)[0])
        elif operator == "<":
            index.extend(np.where(search_data < value)[0])

    elif is_categorical == 1:
        search_data = topo.categorical_data[:, column_index]
        if operator == "=":
            index.extend(np.where(search_data == value)[0])
        elif operator == "like":
            for i in range(search_data.shape[0]):
                if value in search_data[i]:
                    index.append(i)

    for i in index:
        searched_color[i] = colorlist[i]
    return searched_color


def _search(search_params, canvas_params, db_data, color_data):
    canvas_colors = {}
    color_data_avg = np.average(color_data, axis=0)
    color_data_std = np.std(color_data, axis=0)
    operators = ["=", "like"]

    for key in canvas_params.keys():
        # インスタンス初期化
        topology = SearchableTopology(verbose=0)
        _set_canvas_data(topology, db_data[key])
        colors = db_data[key]["colors"]

        # 検索
        if len(search_params["search_value"]) > 0:
            decoded_value = _decode_txt(search_params["search_value"])
            # 数値データなら
            if search_params["is_categorical"] == 0:
                operators = ["=", ">", "<"]
                column_avg = color_data_avg[search_params["search_column_index"]]
                column_std = color_data_std[search_params["search_column_index"]]
                decoded_value = (float(decoded_value) - column_avg) / (column_std + 1e-6)

            if canvas_params[key]["mode"] == 0 or canvas_params[key]["mode"] == 1:
                # point cloudを検索
                colors = _search_node(topology, decoded_value, colors,
                                      search_params["is_categorical"],
                                      operators[search_params["operator_index"]],
                                      search_params["search_column_index"],
                                      color_data[:, search_params["search_column_index"]])
            elif canvas_params[key]["mode"] == 2:
                dict_index = str(search_params["is_categorical"]) + str(search_params["search_column_index"])
                dict_values = {"index": search_params["search_column_index"],
                               "data_type": search_params["is_categorical"],
                               "operator": operators[search_params["operator_index"]],
                               "value": decoded_value}
                # トポロジーを検索
                topology.advanced_search({dict_index: dict_values}, color_data[:, db_data[key]["color_index"]])
                colors = topology.colors

        canvas_colors.update({key: {"colors": colors}})
    return canvas_colors


@route("/api/search", method="POST")
def search():
    rand_str = request.get_cookie("rand_str")
    # パラメータ取得
    filename = request.params.filename
    filepath = os.path.join(DATA_DIR, filename)
    colorize_topology_index = list(map(int, request.params.colorize_topology_index.split(",")))

    search_params = {
        "is_categorical": int(request.params.is_categorical),
        "search_column_index": int(request.params.search_column_index),
        "operator_index": int(request.params.operator_index),
        "search_value": request.params.search_value
    }
    canvas_params = {
        0: {
            "mode": int(request.params.mode),
            "color_index": int(request.params.color_index)
        },
        1: {
            "mode": int(request.params.mode_col2),
            "color_index": int(request.params.color_index_col2)
        }
    }

    db_data = storage.get_topology_data(rand_str)

    # データ読み込み
    pdata, categorical_data_index, numerical_data_index = _get_data_and_index(filepath)
    numerical_data = np.array(pdata.loc[:, numerical_data_index])
    # 色付けに使うデータを抽出
    color_data = numerical_data[:, colorize_topology_index]

    canvas_colors = _search(search_params, canvas_params, db_data, color_data)

    body = json.dumps({"canvas_colors": canvas_colors})
    r = set_json_body(body)
    return r


global storage
storage = Storage()

run(host="0.0.0.0", port=8080)
