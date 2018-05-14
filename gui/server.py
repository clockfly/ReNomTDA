# coding: utf-8
import argparse
import json
import os
import numpy as np
import pandas as pd
import re
import scipy
import string
from bottle import HTTPResponse, default_app, request, route, run, static_file
from sklearn import cluster, ensemble, neighbors, preprocessing, svm
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

import renom as rm
from renom.optimizer import Adam

from renom_tda.lens import PCA, TSNE, Isomap
from renom_tda.topology import Topology
from renom_tda.loader import CSVLoader
from renom_tda.utils import GraphUtil

from storage import storage
from settings import ENCODING
import wsgi_server


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MIN_NODE_SIZE = 3.0
MAX_NODE_SIZE = 15.0

REDUCTIONS = [
    PCA(components=[0, 1]),
    TSNE(components=[0, 1]),
    Isomap(components=[0, 1]),
    None]


def create_response(body):
    # create json response
    r = HTTPResponse(status=200, body=body)
    r.set_header('Content-Type', 'application/json')
    return r


@route('/', method='GET')
def index():
    try:
        # initialize db
        init_db()
        return static_file('index.html', root='js')
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        r = create_response(body)
        return r


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


def _get_file_name_from_id(file_id):
    return storage.fetch_file(file_id)['name']


def _get_number_index_from_dataframe(data):
    return np.logical_or(data.dtypes == "float", data.dtypes == "int")


def _load_data(file_id):
    file_name = _get_file_name_from_id(file_id)
    file_data = pd.read_csv(os.path.join(DATA_DIR, file_name), encoding=ENCODING).dropna()
    return file_data


def _get_number_and_text_data(data):
    # get index of number data
    number_index = _get_number_index_from_dataframe(data)

    # get text data
    text_data = data.loc[:, ~number_index]
    text_columns = text_data.columns

    # get number data
    number_columns = data.loc[:, number_index].columns
    number_data = np.array(data)
    number_data[:, ~number_index] = np.zeros(text_data.shape)
    number_data = number_data.astype('float')

    return text_data, text_columns, number_data, number_columns, number_index


@route('/api/files/<file_id:int>', method='GET')
def load_file(file_id):
    try:
        file_data = _load_data(file_id)
        data_header = file_data.columns

        text_data, text_columns, number_data, number_columns, number_index = _get_number_and_text_data(file_data)

        # transpose number data for histogram
        hist_data = number_data.T
        topo_hist = number_data[:, number_index].T

        # statstic values
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
            'file_data': np.array(file_data).tolist(),
            'number_index': number_index.tolist(),
            'number_columns': list(number_columns),
            'text_columns': list(text_columns),
            'number_data': number_data.tolist(),
            'text_data': np.array(text_data).tolist(),
            'hist_data': hist_data.tolist(),
            'topo_hist': topo_hist.tolist(),
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


def _convert_array_to_hypercubes(hypercubes):
    ret = {}
    for i, v in enumerate(hypercubes):
        ret.update({i: v})
    return ret


def _split_target(data, target_index):
    target = data[:, target_index]
    ret = data[:, np.arange(data.shape[1]) != target_index]
    return ret, target


def _concat_target(data, target, target_index):
    ret = np.concatenate([data[:, :target_index], target.reshape(-1, 1), data[:, target_index:]], axis=1)
    return ret


@route("/api/reduction", method="GET")
def reduction():
    try:
        # get request params
        file_id = request.params.file_id
        target_index = request.params.target_index
        algorithm = int(request.params.algorithm)

        # get file name
        file_name = _get_file_name_from_id(file_id)
        file_path = os.path.join(DATA_DIR, file_name)

        # create topology instance
        topology = Topology(verbose=0)
        loader = CSVLoader(file_path)
        topology.load(loader=loader, standardize=True)

        # If target index isn't exists, use all data to calculate
        if target_index != '':
            topology.number_data, target = _split_target(topology.number_data, int(target_index))

        # transform & scaling data
        scaler = preprocessing.MinMaxScaler(feature_range=(0.05, 0.95))
        topology.fit_transform(lens=[REDUCTIONS[algorithm]], scaler=scaler)

        body = {
            "point_cloud": topology.point_cloud.tolist(),
        }
        r = create_response(body)
        r.set_header('Cache-Control', 'max-age=86400')
        return r

    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        r = create_response(body)
        return r


@route("/api/create", method="POST")
def create():
    try:
        # get request params
        data = json.loads(request.params.data)

        file_id = int(data["file_id"])
        file_name = _get_file_name_from_id(file_id)
        file_path = os.path.join(DATA_DIR, file_name)

        # create topology instance
        topology = Topology(verbose=0)
        loader = CSVLoader(file_path)
        topology.load(loader=loader, standardize=True)

        target_index = data["target_index"]
        mode = int(data["mode"])
        clustering_algorithm = int(data["clustering_algorithm"])
        train_size = float(data["train_size"])
        k = int(data["k"])
        eps = float(data["eps"])
        min_samples = int(data["min_samples"])
        resolution = int(data["resolution"])
        overlap = float(data["overlap"])
        topology.point_cloud = np.array(data["point_cloud"])

        if mode == 0:
            # scatter plot
            colors = []
            for i in range(topology.number_data.shape[1]):
                t = topology.number_data[:, i]
                scaler = preprocessing.MinMaxScaler()
                t = scaler.fit_transform(t.reshape(-1, 1))
                topology.color_point_cloud(target=t, normalize=False)
                colors.append(topology.point_cloud_hex_colors)

        elif mode == 1:
            # unsupervised_clusterings
            # If target index isn't exists, use all data to calculate
            if target_index != '':
                topology.number_data, target = _split_target(topology.number_data, int(target_index))

            clusters = [
                cluster.KMeans(n_clusters=k),
                cluster.DBSCAN(eps=eps, min_samples=min_samples)
            ]
            topology.unsupervised_clustering_point_cloud(clusterer=clusters[clustering_algorithm])

            if target_index != '':
                topology.number_data = _concat_target(topology.number_data, target, int(target_index))

            colors = []
            for i in range(topology.number_data.shape[1]):
                colors.append(topology.point_cloud_hex_colors)

        elif mode == 2:
            # supervised_clusterings
            # If target index isn't exists, use all data to calculate
            if target_index != '':
                topology.number_data, target = _split_target(topology.number_data, int(target_index))

            clusters = [
                neighbors.KNeighborsClassifier(n_neighbors=k)
            ]
            topology.supervised_clustering_point_cloud(clusterer=clusters[clustering_algorithm],
                                                       target=target, train_size=train_size)

            if target_index != '':
                topology.number_data = _concat_target(topology.number_data, target, int(target_index))

            colors = []
            for i in range(topology.number_data.shape[1]):
                colors.append(topology.point_cloud_hex_colors)

        elif mode == 3:
            # tda
            topology.map(resolution=resolution, overlap=overlap, eps=eps, min_samples=min_samples)

            colors = []
            for i in range(topology.number_data.shape[1]):
                t = topology.number_data[:, i]
                scaler = preprocessing.MinMaxScaler()
                t = scaler.fit_transform(t.reshape(-1, 1))
                topology.color(target=t)
                colors.append(topology.hex_colors)

        if mode < 3:
            hypercubes = np.arange(len(topology.point_cloud)).reshape(-1, 1).tolist()
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
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        r = create_response(body)
        return r


@route('/api/export', method='POST')
def export():
    try:
        file_id = request.params.file_id
        file_name = _get_file_name_from_id(file_id)
        file_path = os.path.join(DATA_DIR, file_name)

        # get output filename from request
        out_file_name = request.params.out_file_name
        if not re.match(r'[a-zA-Z0-9]*.csv', out_file_name):
            out_file_name += '.csv'

        click_node_data_ids = np.array(json.loads(request.params.click_node_data_ids))

        pdata = pd.read_csv(file_path)
        out_file_path = os.path.join(DATA_DIR, out_file_name)
        pdata.iloc[click_node_data_ids].to_csv(out_file_path, index=False)
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        r = create_response(body)
        return r


@route('/api/search', method='POST')
def search():
    try:
        data = json.loads(request.params.data)

        topology = Topology(verbose=0)
        file_id = int(data["file_id"])
        file_name = _get_file_name_from_id(file_id)
        file_path = os.path.join(DATA_DIR, file_name)

        # create topology instance
        topology = Topology(verbose=0)
        loader = CSVLoader(file_path)
        topology.load(loader=loader, standardize=True)

        target_index = data["target_index"]
        mode = int(data["mode"])
        clustering_algorithm = int(data["clustering_algorithm"])
        train_size = float(data["train_size"])
        k = int(data["k"])
        eps = float(data["eps"])
        min_samples = int(data["min_samples"])
        topology.point_cloud = np.array(data["point_cloud"])
        hypercubes = data["hypercubes"]
        topology.hypercubes = _convert_array_to_hypercubes(hypercubes)
        topology.nodes = np.array(data["nodes"])
        topology.edges = np.array(data["edges"])
        search_type = data["search_type"]
        search_conditions = data["search_conditions"]

        colors = []
        if mode == 0:
            for i in range(topology.number_data.shape[1]):
                t = topology.number_data[:, i]
                scaler = preprocessing.MinMaxScaler()
                t = scaler.fit_transform(t.reshape(-1, 1))
                topology.color_point_cloud(target=t)
                if len(search_conditions) > 0:
                    topology.search_point_cloud(search_dicts=search_conditions, search_type=search_type)
                colors.append(topology.point_cloud_hex_colors)
        elif mode == 1:
            # unsupervised_clusterings
            if target_index != '':
                topology.number_data, target = _split_target(topology.number_data, int(target_index))

            clusters = [
                cluster.KMeans(n_clusters=k),
                cluster.DBSCAN(eps=eps, min_samples=min_samples)
            ]
            topology.unsupervised_clustering_point_cloud(clusterer=clusters[clustering_algorithm])

            if target_index != '':
                topology.number_data = _concat_target(topology.number_data, target, int(target_index))

            if len(search_conditions) > 0:
                topology.search_point_cloud(search_dicts=search_conditions, search_type=search_type)

            colors = []
            for i in range(topology.number_data.shape[1]):
                colors.append(topology.point_cloud_hex_colors)

        elif mode == 2:
            # supervised_clusterings
            if target_index != '':
                topology.number_data, target = _split_target(topology.number_data, int(target_index))

            clusters = [
                neighbors.KNeighborsClassifier(n_neighbors=k)
            ]
            topology.supervised_clustering_point_cloud(clusterer=clusters[clustering_algorithm],
                                                       target=target, train_size=train_size)

            if target_index != '':
                topology.number_data = _concat_target(topology.number_data, target, int(target_index))

            if len(search_conditions) > 0:
                topology.search_point_cloud(search_dicts=search_conditions, search_type=search_type)

            colors = []
            for i in range(topology.number_data.shape[1]):
                colors.append(topology.point_cloud_hex_colors)

        elif mode == 3:
            # tda
            topology.graph_util = GraphUtil(point_cloud=topology.point_cloud, hypercubes=topology.hypercubes)

            colors = []
            for i in range(topology.number_data.shape[1]):
                t = topology.number_data[:, i]
                scaler = preprocessing.MinMaxScaler()
                t = scaler.fit_transform(t.reshape(-1, 1))
                topology.color(target=t)
                if len(search_conditions) > 0:
                    topology.search(search_dicts=search_conditions, search_type=search_type)
                colors.append(topology.hex_colors)


        body = {
            "colors": colors,
        }
        r = create_response(body)
        return r

    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        r = create_response(body)
        return r


def init_db():
    files = os.listdir(DATA_DIR)
    files_in_db = storage.fetch_files()

    # file name list in storage
    name_list = []
    for i in files_in_db:
        name_list.append(files_in_db[i]['name'])

    # register file
    for f in files:
        if not storage.exist_file(f):
            storage.register_file(f)

    # remove file
    diff_files = list(set(name_list) - set(files))
    for f in diff_files:
        storage.remove_file(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='desc')
    parser.add_argument('--host', default='0.0.0.0', help='Server address')
    parser.add_argument('--port', default='8080', help='Server port')
    args = parser.parse_args()

    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    init_db()

    wsgiapp = default_app()
    httpd = wsgi_server.Server(wsgiapp, host=args.host, port=int(args.port))
    httpd.serve_forever()
