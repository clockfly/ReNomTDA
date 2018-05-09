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

from renom_tda.lens import PCA, TSNE, Isomap
from renom_tda.lens_renom import AutoEncoder
from renom_tda.topology import Topology
from renom_tda.loader import CSVLoader
from renom_tda.utils import GraphUtil

from storage import storage
from settings import ENCODING

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MIN_NODE_SIZE = 3.0
MAX_NODE_SIZE = 15.0

DATA_TYPE = ["number", "text"]
OPERATORS = [["=", ">", "<"], ["=", "like"]]

REDUCTIONS = [
    PCA(components=[0, 1]),
    TSNE(components=[0, 1]),
    Isomap(components=[0, 1]),
    None]


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
        hist_data = number_data.T

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
            'number_data': number_data.tolist(),
            'hist_data': hist_data.tolist(),
            'data_mean': data_mean.tolist(),
            'data_var': data_var.tolist(),
            'data_std': data_std.tolist(),
            'data_min': data_min.tolist(),
            'data_25percentile': data_25percentile.tolist(),
            'data_50percentile': data_50percentile.tolist(),
            'data_75percentile': data_75percentile.tolist(),
            'data_max': data_max.tolist(),
        })
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
    r = create_response(body)
    return r


def _convert_hypercubes_to_array(hypercubes):
    ret = []
    for i in range(len(hypercubes)):
        ret.append([int(x) for x in hypercubes[i]])
    return ret


@route("/api/create", method="POST")
def create():
    # get request params
    file_id = request.params.file_id
    target_index = request.params.target_index
    algorithm = int(request.params.algorithm)
    mode = int(request.params.mode)
    clustering_algorithm = int(request.params.clustering_algorithm)
    train_size = float(request.params.train_size)
    k = int(request.params.k)
    eps = float(request.params.eps)
    min_samples = int(request.params.min_samples)
    resolution = int(request.params.resolution)
    overlap = float(request.params.overlap)
    color_index = int(request.params.color_index)

    # get file name
    file_name = get_file_name_from_id(file_id)
    file_path = os.path.join(DATA_DIR, file_name)

    # create topology instance
    topology = Topology(verbose=0)
    loader = CSVLoader(file_path)
    topology.load(loader=loader, standardize=True)

    if target_index == '':
        target = topology.number_data[:, 0]
    else:
        i = int(target_index)
        target = topology.number_data[:, i]
        topology.number_data = topology.number_data[:, np.arange(topology.number_data.shape[1]) != i]

    scaler = preprocessing.MinMaxScaler(feature_range=(0.01, 0.99))
    topology.fit_transform(lens=[REDUCTIONS[algorithm]], scaler=scaler)

    if mode == 0:
        colors = []
        for i in range(topology.number_data.shape[1]):
            t = topology.number_data[:, i]
            scaler = preprocessing.MinMaxScaler()
            t = scaler.fit_transform(t.reshape(-1, 1))
            topology.color_point_cloud(target=t, normalize=False)
            colors.append(topology.point_cloud_hex_colors)

        if target_index != '':
            scaler = preprocessing.MinMaxScaler()
            target = scaler.fit_transform(target.reshape(-1, 1))
            topology.color_point_cloud(target=target, normalize=False)
            colors.append(topology.point_cloud_hex_colors)

    elif mode == 1:
        clusters = [
            cluster.KMeans(n_clusters=k),
            cluster.DBSCAN(eps=eps, min_samples=min_samples)
        ]
        topology.unsupervised_clustering_point_cloud(clusterer=clusters[clustering_algorithm])

        if target_index != '':
            columns = topology.number_data.shape[1]+1
        else:
            columns = topology.number_data.shape[1]
        colors = []
        for i in range(columns):
            colors.append(topology.point_cloud_hex_colors)

    elif mode == 2:
        clusters = [
            neighbors.KNeighborsClassifier(n_neighbors=k)
        ]
        topology.supervised_clustering_point_cloud(clusterer=clusters[clustering_algorithm], target=target)

        if target_index != '':
            columns = topology.number_data.shape[1]+1
        else:
            columns = topology.number_data.shape[1]
        colors = []
        for i in range(columns):
            colors.append(topology.point_cloud_hex_colors)

    elif mode == 3:
        topology.map(resolution=resolution, overlap=overlap, eps=eps, min_samples=min_samples)

        colors = []
        for i in range(topology.number_data.shape[1]):
            t = topology.number_data[:, i]
            scaler = preprocessing.MinMaxScaler()
            t = scaler.fit_transform(t.reshape(-1, 1))
            topology.color(target=t)
            colors.append(topology.hex_colors)

        if target_index != '':
            topology.color(target=target)
            colors.append(topology.hex_colors)

    if mode < 3:
        hypercubes = np.arange(len(topology.point_cloud)).reshape(-1,1).tolist()
        nodes = topology.point_cloud.tolist()
        edges = []
        node_sizes = [MIN_NODE_SIZE] * len(topology.point_cloud)
        colors = colors
    elif mode == 3:
        hypercubes = _convert_hypercubes_to_array(topology.hypercubes)
        nodes = topology.nodes.tolist()
        edges = topology.edges.tolist()
        scaler = preprocessing.MinMaxScaler(feature_range=(MIN_NODE_SIZE, MAX_NODE_SIZE))
        node_sizes = scaler.fit_transform(topology.node_sizes).tolist()
        colors = colors

    body = {
        "hypercubes": hypercubes,
        "nodes": nodes,
        "edges": edges,
        "node_sizes": node_sizes,
        "colors": colors,
        "train_index": topology.train_index.tolist()
    }
    r = create_response(body)
    return r


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
