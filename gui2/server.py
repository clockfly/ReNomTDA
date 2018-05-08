# coding: utf-8
import argparse
import json
import os
import numpy as np
import pandas as pd
import pkg_resources
import random
import scipy
import string
from bottle import HTTPResponse, request, route, run, static_file
from sklearn import cluster, ensemble, neighbors, preprocessing, svm
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

import renom as rm
from renom.optimizer import Adam

from renom_tda.lens import PCA, TSNE, MDS, Isomap
from renom_tda.lens_renom import AutoEncoder
from renom_tda.topology import Topology
from renom_tda.utils import GraphUtil

from storage import storage
from settings import ENCODING

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MIN_NODE_SIZE = 0.3
MAX_NODE_SIZE = 2.0

DATA_TYPE = ["number", "text"]
OPERATORS = [["=", ">", "<"], ["=", "like"]]


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


def create_response(body):
    # httpのjsonレスポンスを作成する
    r = HTTPResponse(status=200, body=body)
    r.set_header('Content-Type', 'application/json')
    return r


@route('/', method='GET')
def index():
    return static_file('index.html', root='js')


# @route("/api/set_random", method="POST")
# def set_random():
#     rand_str = request.params.rand_str
#     if rand_str == "":
#         cookie_str = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(20)])
#
#         storage.regist_topology_data(cookie_str, 0)
#         storage.regist_topology_data(cookie_str, 1)
#
#         body = json.dumps({"rand_str": cookie_str})
#         r = set_json_body(body)
#         return r
#     return


# @route("/api/reset")
# def reset():
#     return


@route("/build/<file_name:path>")
def static(file_name):
    return static_file(file_name, root='js/build/', mimetype='application/javascript')

@route('/static/js/<file_name:path>')
def static_js(file_name):
    return static_file(file_name, root='js/static/js/')


@route('/static/css/<file_name:path>')
def static_css(file_name):
    return static_file(file_name, root='js/static/css/', mimetype='text/css')


@route('/static/fonts/<file_name:path>')
def static_fonts(file_name):
    return static_file(file_name, root='js/static/fonts/')


@route("/api/files", method="GET")
def load_files():
    try:
        files = storage.fetch_files()
        body = json.dumps({"files": files})
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
    r = create_response(body)
    return r


def get_file_name_from_id(file_id):
    # ファイルIDからファイル名を取得する
    return storage.fetch_file(file_id)['name']


def get_number_index_from_dataframe(data):
    # pandasデータフレームの数値データの列のインデックスを取得する
    return np.logical_or(data.dtypes == "float", data.dtypes == "int")


def get_hist_from_npdata(number_index, data):
    # ヒストグラム用にNone値を削除した配列の配列を作成する
    hist_data = []
    for i, val in enumerate(number_index):
        if val:
            # 数値データのindexにはNone値を削除した配列を入れる
            d = data[:, i][~np.isnan(data[:, i])]
            hist_data.append(d.tolist())
        else:
            # 文字列データのindexには空の配列を入れる
            hist_data.append([])
    return hist_data


@route('/api/files/<file_id:int>', method='GET')
def load_file(file_id):
    try:
        # ファイルIDからファイル名を取得
        file_name = get_file_name_from_id(file_id)

        # ファイルを設定したエンコードで読み込む
        file_data = pd.read_csv(os.path.join(DATA_DIR, file_name), encoding=ENCODING)

        # ヘッダー
        data_header = file_data.columns

        # 数値データのカラムのインデックスを取得する
        number_index = get_number_index_from_dataframe(file_data)

        # 文字列データを取得
        text_data = file_data.loc[:, ~number_index]

        # 数値データを取得
        # 配列の大きさが変更されないように文字列データの部分は0で埋めておく
        number_data = np.array(file_data)
        number_data[:, ~number_index] = np.zeros(text_data.shape)
        number_data = number_data.astype('float')

        # ヒストグラム表示用のデータを作成
        hist_data = get_hist_from_npdata(number_index, number_data)

        # # 欠損のインデックスを取得
        # nan_index = file_data.isnull()
        # # 欠損の数と率を取得
        # nan_count = nan_index.sum().astype(np.float)
        # nan_ratio = (nan_count / file_data.shape[0]).astype(np.float)
        # interpolate_list = np.zeros(nan_index.shape[1])

        # 統計量を取得
        data_mean = np.nanmean(number_data, axis=0)
        data_var = np.nanvar(number_data, axis=0)
        data_std = np.nanstd(number_data, axis=0)
        data_min = np.nanmin(number_data, axis=0)
        data_25percentile = np.nanpercentile(number_data, 25, axis=0)
        data_50percentile = np.nanpercentile(number_data, 50, axis=0)
        data_75percentile = np.nanpercentile(number_data, 75, axis=0)
        data_max = np.nanmax(number_data, axis=0)

        body = json.dumps({
            'row': file_data.shape[0],
            'columns': file_data.shape[1],
            'data_header': data_header.tolist(),
            'number_index': number_index.tolist(),
            'hist_data': hist_data,
            'data_mean': data_mean.tolist(),
            'data_var': data_var.tolist(),
            'data_std': data_std.tolist(),
            'data_min': data_min.tolist(),
            'data_25percentile': data_25percentile.tolist(),
            'data_50percentile': data_50percentile.tolist(),
            'data_75percentile': data_75percentile.tolist(),
            'data_max': data_max.tolist(),
            # 'nan_count': nan_count.tolist(),
            # 'nan_ratio': nan_ratio.tolist(),
        })
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
    r = create_response(body)
    return r

# @route("/static/css/<file_name>")
# def css(file_name):
#     return static_file(file_name, root=os.path.join(BASE_DIR, 'static', 'css'))
#
#
# @route("/static/<file_name>")
# def static(file_name):
#     return pkg_resources.resource_string(__name__, "static/" + file_name)
#
#
# @route("/fonts/<file_name>")
# def fonts(file_name):
#     return pkg_resources.resource_string(__name__, "static/fonts/" + file_name)


# def _get_data_and_index(filepath):
#     # nanを含むデータは使わない
#     pdata = pd.read_csv(filepath)
#     # 文字列データと数値データを分ける
#     categorical_data_index = (pdata.dtypes == "object")
#     numerical_data_index = np.logical_or(pdata.dtypes == "float", pdata.dtypes == "int")
#     return pdata, categorical_data_index, numerical_data_index
#
#
# @route("/api/load_file_list", method="GET")
# def load_file_list():
#     files = os.listdir(DATA_DIR)
#     files_in_db = storage.get_files()
#
#     name_list = []
#     for i in files_in_db:
#         name_list.append(files_in_db[i]["name"])
#
#     # check increase file
#     diff_files = list(set(files) - set(name_list))
#     for f in diff_files:
#         storage.register_file(f)
#
#     # check decrease file
#     diff_files = list(set(name_list) - set(files))
#     for f in diff_files:
#         storage.remove_file(f)
#
#     files_in_db = storage.get_files()
#     body = json.dumps({"files": files_in_db})
#     r = set_json_body(body)
#     return r
#
#
# # fileのロード
# # TODO
# # ディレクトリ決め打ちを変更
# @route("/api/load_file", method="POST")
# def load_file():
#     file_id = request.params.file_id
#     filename = storage.get_file_name(file_id)
#     filepath = os.path.join(DATA_DIR, filename)
#
#     try:
#         pdata, categorical_data_index, numerical_data_index = _get_data_and_index(filepath)
#
#         if pdata.shape[0] > 10000:
#             body = json.dumps({"error": "LF0002"})
#             r = set_json_body(body)
#             return r
#
#         # 表示するカラム名や統計情報を取得する
#         labels = pdata.columns
#         categorical_data_labels = labels[categorical_data_index]
#         # 数値データのラベル
#         numerical_data_labels = labels[numerical_data_index]
#
#         # 数値データはトポロジーの作成に使う項目と色をつける項目の絞り込みで使う
#         numerical_data = np.array(pdata.loc[:, numerical_data_index])
#
#         # 統計情報を取得する
#         numerical_data_mins = numerical_data.min(axis=0)
#         numerical_data_mins = np.around(numerical_data_mins, 2)
#         numerical_data_maxs = numerical_data.max(axis=0)
#         numerical_data_maxs = np.around(numerical_data_maxs, 2)
#         numerical_data_means = numerical_data.mean(axis=0)
#         numerical_data_means = np.around(numerical_data_means, 2)
#
#         body = json.dumps({"numerical_data": np.sort(numerical_data.T).tolist(),
#                            "categorical_data_labels": categorical_data_labels.tolist(),
#                            "numerical_data_labels": numerical_data_labels.tolist(),
#                            "numerical_data_mins": numerical_data_mins.tolist(),
#                            "numerical_data_maxs": numerical_data_maxs.tolist(),
#                            "numerical_data_means": numerical_data_means.tolist()})
#         r = set_json_body(body)
#         return r
#
#     except IOError:
#         body = json.dumps({"error": "LF0001"})
#         r = set_json_body(body)
#         return r
#
#
# # ノードをクリックした時に呼び出す関数
# @route("/api/click", method="POST")
# def click():
#     rand_str = request.params.rand_str
#
#     file_id = request.params.file_id
#     filename = storage.get_file_name(file_id)
#     filepath = os.path.join(DATA_DIR, filename)
#     node_index = int(request.params.clicknode)
#     columns = int(request.params.columns)
#     mode = int(request.params.mode)
#
#     db_data = storage.get_topology_data(rand_str)
#
#     topology = Topology(verbose=0)
#     _set_canvas_data(topology, db_data[columns])
#
#     # データ読み込み
#     pdata, categorical_data_index, numerical_data_index = _get_data_and_index(filepath)
#     numerical_data = np.array(pdata.loc[:, numerical_data_index])
#     categorical_data = np.array(pdata.loc[:, categorical_data_index])
#
#     if mode == 0 or mode == 1 or mode == 2:
#         numerical_data = np.around(numerical_data[node_index], 2)
#         body = json.dumps({"categorical_data": [categorical_data[node_index, :].tolist()],
#                            "data": [numerical_data.tolist()]})
#     elif mode == 3:
#         categorical_data = categorical_data[topology.hypercubes[node_index]]
#         numerical_data = numerical_data[topology.hypercubes[node_index]]
#         # 少数の桁を丸める
#         numerical_data = np.around(numerical_data, 2)
#
#         body = json.dumps({"categorical_data": categorical_data.tolist(), "data": numerical_data.tolist()})
#
#     r = set_json_body(body)
#     return r
#
#
# def _dimension_reduction(topology, params, calc_data):
#     algorithms = [PCA(components=[0, 1]),
#                   TSNE(components=[0, 1]),
#                   MDS(components=[0, 1]),
#                   Isomap(components=[0, 1]),
#                   AutoEncoder(epoch=200,
#                               batch_size=100,
#                               network=AutoEncoder2Layer(calc_data.shape[1]),
#                               opt=Adam()),
#                   None]
#     # 表示が切れるので、0~1ではなく0.01~0.99に正規化
#     scaler = preprocessing.MinMaxScaler(feature_range=(0.01, 0.99))
#     if algorithms[params["algorithm"]] is None:
#         topology.point_cloud = topology.number_data[:, :2]
#         print(topology.point_cloud.shape)
#     else:
#         topology.fit_transform(metric=None, lens=[
#                                algorithms[params["algorithm"]]], scaler=scaler)
#
#
# def _get_topology_color(topo, cdata):
#     # トポロジーの色の設定
#     topo.color(cdata)
#     return topo.hex_colors
#
#
# def _get_topology_sizes(topo, resolution):
#     # トポロジーの大きさの調整
#     max_scale = 80 / resolution
#     if max_scale > MAX_NODE_SIZE:
#         max_scale = MAX_NODE_SIZE
#     scaler = preprocessing.MinMaxScaler(feature_range=(MIN_NODE_SIZE, max_scale))
#     topo.node_sizes = scaler.fit_transform(topo.node_sizes)
#     return topo.node_sizes.tolist()
#
#
# def _check_file_and_algorithm(db_data, params, filename):
#     if db_data["filename"] != filename:
#         return True
#     if db_data["algorithm"] != params["algorithm"]:
#         return True
#     return False
#
#
# def _check_analysis_mode(db_data, params):
#     if db_data["mode"] != params["mode"]:
#         return True
#     return False
#
#
# def _check_params_change(db_data, params, filename):
#     if db_data["filename"] != filename:
#         return True
#     if db_data["resolution"] != params["resolution"]:
#         return True
#     if db_data["overlap"] != params["overlap"]:
#         return True
#     if db_data["eps"] != params["eps"]:
#         return True
#     if db_data["min_samples"] != params["min_samples"]:
#         return True
#     return False
#
#
# def _calc_dist_vec(topology):
#     # 距離のベクトル(小さい順)を返す
#     scaler = preprocessing.MinMaxScaler()
#     scaled_data = scaler.fit_transform(topology.number_data)
#     dist_mat = scipy.spatial.distance.cdist(scaled_data, scaled_data)
#     dist_vec = np.trim_zeros(np.unique(dist_mat))
#     return dist_vec
#
#
# def _get_canvas_data_from_db_data(db_data):
#     nodes = db_data["nodes"]
#     edges = db_data["edges"]
#     sizes = db_data["sizes"]
#     return nodes, edges, sizes
#
#
# def _set_topology_data(topology, db_data):
#     topology.load_data(number_data=db_data["input_data"], standardize=True)
#     topology.point_cloud = db_data["point_cloud"]
#     topology.hypercubes = db_data["hypercubes"]
#     topology.graph = GraphUtil(point_cloud=db_data["point_cloud"], hypercubes=db_data["hypercubes"])
#
#
# def _create(rand_str, canvas_params, calc_data, color_data, categorical_data, db_data, filename):
#     canvas_data = {}
#     pca_result = {
#         "axis": [],
#         "contribution_ratio": 0,
#         "top_index": []
#     }
#
#     for key in canvas_params.keys():
#         # インスタンス初期化
#         topology = Topology(verbose=0)
#         topology.load_data(number_data=calc_data, text_data=categorical_data, standardize=True)
#
#         if _check_file_and_algorithm(db_data[key], canvas_params[key], filename):
#             # アルゴリズムが変わっていたら次元削減
#             _dimension_reduction(topology, canvas_params[key], calc_data)
#             # pcaの主成分軸を設定
#             if canvas_params[key]["algorithm"] == 0:
#                 sort_index = np.argsort(np.abs(topology.lens[0].components_), axis=1)
#                 pca_result["axis"] = np.around(topology.lens[0].components_, 3).tolist()
#                 pca_result["contribution_ratio"] = np.around(np.sum(topology.lens[0].explained_variance_ratio_), 3)
#                 pca_result["top_index"] = sort_index[:, -3:].tolist()
#         else:
#             # アルゴリズムが変わっていなければDBの値を使う
#             _set_topology_data(topology, db_data[key])
#             pca_result = db_data[key]["pca_result"]
#
#         cdata = color_data[:, canvas_params[key]["color_index"]].reshape(-1, 1)
#
#         # scatter plot
#         if canvas_params[key]["mode"] == 0:
#             nodes = topology.point_cloud.tolist()
#             edges = []
#             sizes = [MIN_NODE_SIZE] * len(topology.point_cloud)
#             topology.color_point_cloud(cdata)
#             colors = topology.point_cloud_hex_colors
#         # unsupervised clustering
#         elif canvas_params[key]["mode"] == 1:
#             dist_vec = _calc_dist_vec(topology)
#             eps = dist_vec.min() + (dist_vec.max() - dist_vec.min()) * canvas_params[key]["eps"]
#             clusterers = [cluster.KMeans(n_clusters=canvas_params[key]["class_count"]),
#                           cluster.DBSCAN(eps=eps, min_samples=canvas_params[key]["min_samples"])]
#             clusterer = clusterers[canvas_params[key]["clustering_index"]]
#             topology.unsupervised_clustering_point_cloud(clusterer=clusterer)
#
#             nodes = topology.point_cloud.tolist()
#             edges = []
#             sizes = [MIN_NODE_SIZE] * len(topology.point_cloud)
#             colors = topology.point_cloud_hex_colors
#         # supervised clustering
#         elif canvas_params[key]["mode"] == 2:
#             clusterers = [neighbors.KNeighborsClassifier(n_neighbors=canvas_params[key]["neighbors"]),
#                           svm.SVC(),
#                           ensemble.RandomForestClassifier()]
#             clusterer = clusterers[canvas_params[key]["clustering_index"]]
#             topology.supervised_clustering_point_cloud(clusterer=clusterer, target=cdata,
#                                                        train_size=canvas_params[key]["train_size"])
#
#             nodes = topology.point_cloud.tolist()
#             edges = []
#             sizes = [MIN_NODE_SIZE] * len(topology.point_cloud)
#             colors = topology.point_cloud_hex_colors
#         # TDA
#         elif canvas_params[key]["mode"] == 3:
#             # パラメータが変わっていたらトポロジーの再計算
#             check_file_and_algorithm = _check_file_and_algorithm(db_data[key], canvas_params[key], filename)
#             check_analysis_mode = _check_analysis_mode(db_data[key], canvas_params[key])
#             check_params = _check_params_change(db_data[key], canvas_params[key], filename)
#
#             if check_file_and_algorithm or check_analysis_mode or check_params:
#                 topology.map(resolution=canvas_params[key]["resolution"],
#                              overlap=canvas_params[key]["overlap"],
#                              eps=canvas_params[key]["eps"],
#                              min_samples=canvas_params[key]["min_samples"])
#                 # nodeが作れなかったときreturnする
#                 if topology.nodes is None or len(topology.nodes) < 5:
#                     canvas_data.update({key: {"nodes": [], "edges": [], "colors": [], "sizes": []}})
#                     return canvas_data, pca_result
#
#                 nodes = topology.nodes.tolist()
#
#                 if topology.edges is None:
#                     edges = []
#                 else:
#                     edges = topology.edges.tolist()
#
#                 sizes = _get_topology_sizes(topology, canvas_params[key]["resolution"])
#             # パラメータが変わっていなければdbのデータを使う
#             else:
#                 nodes, edges, sizes = _get_canvas_data_from_db_data(db_data[key])
#             colors = _get_topology_color(topology, cdata)
#
#         storage.update_topology_data(rand_str, key, canvas_params, topology, categorical_data,
#                                      nodes, edges, colors, sizes, filename, pca_result)
#
#         canvas_data.update({key: {
#             "nodes": nodes,
#             "edges": edges,
#             "colors": colors,
#             "sizes": sizes,
#             "train_index": topology.train_index.tolist()
#         }})
#
#     return canvas_data, pca_result
#
#
# def _get_statistic_value(data):
#     q1 = scipy.stats.scoreatpercentile(data, 25)
#     median = np.median(data)
#     q3 = scipy.stats.scoreatpercentile(data, 75)
#     return q1, median, q3
#
#
# def _set_histogram_data(color_data, canvas_params):
#     histogram_data = {}
#     for key in canvas_params.keys():
#         cdata = color_data[:, canvas_params[key]["color_index"]].reshape(-1, 1)
#         data_color_values = cdata.reshape(-1,).tolist()
#         q1, median, q3 = _get_statistic_value(cdata)
#         statistic_value = [q1, median, q3]
#         histogram_data.update({key: {
#             "data_color_values": data_color_values,
#             "statistic_value": statistic_value
#         }})
#     return histogram_data
#
#
# # トポロジを作成する関数
# @route("/api/create", method="POST")
# def create():
#     # rand_str = request.get_cookie("rand_str")
#     rand_str = request.params.rand_str
#
#     # パラメータ取得
#     file_id = request.params.file_id
#     filename = storage.get_file_name(file_id)
#     filepath = os.path.join(DATA_DIR, filename)
#     # トポロジーを計算する項目と色をつける項目のインデックスのリストを受け取る
#     create_topology_index = list(map(int, request.params.create_topology_index.split(",")))
#     colorize_topology_index = list(map(int, request.params.colorize_topology_index.split(",")))
#
#     # canvas params
#     # モード 0:scatter plot, 1:Clustering, 2:TDA
#     canvas_params = {
#         0: {
#             "mode": int(request.params.mode),
#             "color_index": int(request.params.color_index),
#             "algorithm": int(request.params.algorithm),
#             "resolution": int(request.params.resolution),
#             "overlap": float(request.params.overlap),
#             "clustering_index": int(request.params.clustering_index),
#             "class_count": int(request.params.class_count),
#             "eps": float(request.params.eps),
#             "min_samples": float(request.params.min_samples),
#             "train_size": float(request.params.train_size),
#             "neighbors": int(request.params.neighbors),
#         },
#         1: {
#             "mode": int(request.params.mode_col2),
#             "color_index": int(request.params.color_index_col2),
#             "algorithm": int(request.params.algorithm_col2),
#             "resolution": int(request.params.resolution_col2),
#             "overlap": float(request.params.overlap_col2),
#             "clustering_index": int(request.params.clustering_index_col2),
#             "class_count": int(request.params.class_count_col2),
#             "eps": float(request.params.eps_col2),
#             "min_samples": float(request.params.min_samples_col2),
#             "train_size": float(request.params.train_size_col2),
#             "neighbors": int(request.params.neighbors_col2),
#         }
#     }
#
#     db_data = storage.get_topology_data(rand_str)
#
#     # データ読み込み
#     pdata, categorical_data_index, numerical_data_index = _get_data_and_index(filepath)
#     numerical_data = np.array(pdata.loc[:, numerical_data_index])
#
#     # 色付けに使うデータを抽出
#     color_data = numerical_data[:, colorize_topology_index]
#     histogram_data = _set_histogram_data(color_data, canvas_params)
#
#     # トポロジーの作成に使うデータを抽出
#     calc_data = numerical_data[:, create_topology_index]
#     # カテゴリデータを抽出
#     categorical_data = np.array(pdata.loc[:, categorical_data_index])
#
#     # トポロジーを作る
#     canvas_data, pca_result = _create(rand_str, canvas_params, calc_data,
#                                       color_data, categorical_data, db_data, filename)
#
#     # nodeが作れなかったときにエラーを返す
#     if len(canvas_data[0]["nodes"]) < 5 or len(canvas_data[1]["nodes"]) < 5:
#         body = json.dumps({"error": True})
#     else:
#         # bodyを設定
#         body = json.dumps({"canvas_data": canvas_data,
#                            "histogram_data": histogram_data,
#                            "pca_result": pca_result
#                            })
#     r = set_json_body(body)
#     return r
#
#
# def _decode_txt(txt):
#     # postデータのバイナリから元の文字列を復元
#     # TODO
#     bytes_str = b""
#     for i in txt.split(","):
#         bytes_str += int(i).to_bytes(1, byteorder='big')
#     decoded_txt = bytes_str.decode('utf-8')
#     return decoded_txt
#
#
# def _set_canvas_data(topology, db_data):
#     topology.load_data(number_data=db_data["input_data"], text_data=db_data["categorical_data"], standardize=True)
#     topology.point_cloud = db_data["point_cloud"]
#     topology.hypercubes = db_data["hypercubes"]
#     topology.nodes = db_data["nodes"]
#     topology.edges = db_data["edges"]
#     topology.hex_colors = db_data["colors"]
#     topology.sizes = db_data["sizes"]
#
#
# def _search(search_params, canvas_params, db_data, color_data, categorical_data):
#     canvas_colors = {}
#
#     for key in canvas_params.keys():
#         # インスタンス初期化
#         topology = Topology(verbose=0)
#         _set_canvas_data(topology, db_data[key])
#         colors = db_data[key]["colors"]
#
#         # 検索
#         if len(search_params["search_value"]) > 0:
#             data_type = DATA_TYPE[search_params["is_categorical"]]
#             operators = OPERATORS[search_params["is_categorical"]]
#             decoded_value = _decode_txt(search_params["search_value"])
#
#             # 数値データなら
#             if search_params["is_categorical"] == 0:
#                 decoded_value = float(decoded_value)
#
#             dict_values = {"column": search_params["search_column_index"],
#                            "data_type": data_type,
#                            "operator": operators[search_params["operator_index"]],
#                            "value": decoded_value}
#
#             cdata = color_data[:, db_data[key]["color_index"]]
#             topology.load_data(color_data, text_data=categorical_data, standardize=True)
#             if canvas_params[key]["mode"] == 3:
#                 # トポロジーを検索
#                 topology.search_from_values(search_dicts=[dict_values], target=None, search_type="and")
#                 colors = topology.hex_colors
#
#             else:
#                 # point cloudを検索
#                 topology.color_point_cloud(cdata)
#                 topology.search_point_cloud(search_dicts=[dict_values], target=None, search_type="and")
#                 colors = topology.point_cloud_hex_colors
#
#         canvas_colors.update({key: {"colors": colors}})
#     return canvas_colors
#
#
# @route("/api/search", method="POST")
# def search():
#     rand_str = request.params.rand_str
#
#     # パラメータ取得
#     file_id = request.params.file_id
#     filename = storage.get_file_name(file_id)
#     filepath = os.path.join(DATA_DIR, filename)
#     colorize_topology_index = list(map(int, request.params.colorize_topology_index.split(",")))
#
#     search_params = {
#         "is_categorical": int(request.params.is_categorical),
#         "search_column_index": int(request.params.search_column_index),
#         "operator_index": int(request.params.operator_index),
#         "search_value": request.params.search_value
#     }
#     canvas_params = {
#         0: {
#             "mode": int(request.params.mode),
#             "color_index": int(request.params.color_index)
#         },
#         1: {
#             "mode": int(request.params.mode_col2),
#             "color_index": int(request.params.color_index_col2)
#         }
#     }
#
#     db_data = storage.get_topology_data(rand_str)
#
#     # データ読み込み
#     pdata, categorical_data_index, numerical_data_index = _get_data_and_index(filepath)
#     numerical_data = np.array(pdata.loc[:, numerical_data_index])
#     # 色付けに使うデータを抽出
#     color_data = numerical_data[:, colorize_topology_index]
#     categorical_data = pdata.loc[:, categorical_data_index]
#
#     canvas_colors = _search(search_params, canvas_params, db_data, color_data, categorical_data)
#
#     body = json.dumps({"canvas_colors": canvas_colors})
#     r = set_json_body(body)
#     return r


class DataFileEventHandler(PatternMatchingEventHandler):
    def on_created(self, event):
        # ファイルが作成されたら
        file_name = event.src_path.split('/')[-1]
        storage.register_file(file_name)

    def on_deleted(self, event):
        # ファイルが削除されたら
        file_name = event.src_path.split('/')[-1]
        storage.remove_file(file_name)


def init_db():
    files = os.listdir(DATA_DIR)
    files_in_db = storage.fetch_files()

    # DB内のファイルの名前のリスト
    name_list = []
    for i in files_in_db:
        name_list.append(files_in_db[i]['name'])

    # DBに登録されていないファイルがあれば登録
    for f in files:
        if not storage.exist_file(f):
            storage.register_file(f)

    # dataディレクトリから削除されたファイルがあればDBに反映
    diff_files = list(set(name_list) - set(files))
    for f in diff_files:
        storage.remove_file(f)


if __name__ == '__main__':
    # 引数でホストとポートを変更できる
    parser = argparse.ArgumentParser(description='desc')
    parser.add_argument('--host', default='0.0.0.0', help='Server address')
    parser.add_argument('--port', default='8080', help='Server port')
    args = parser.parse_args()

    # dataディレクトリの中身をDBに反映
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    init_db()

    # dataディレクトリの中をwatchして変更があったらDBを更新
    handler = DataFileEventHandler(patterns=['*.csv'])
    observer = Observer()
    observer.schedule(handler, DATA_DIR, recursive=False)
    observer.start()

    run(host=args.host, port=args.port, reloader=True)
    observer.stop()
    observer.join()
