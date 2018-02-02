import _pickle as cPickle

import datetime

import os

import sqlite3


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
        c.execute("""CREATE TABLE IF NOT EXISTS file
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         name TEXT NOT NULL,
                         created TIMESTAMP,
                         updated TIMESTAMP,
                         UNIQUE(name))
          """)

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

    def register_file(self, name):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                    INSERT INTO file(name, created, updated)
                    VALUES (?, ?, ?)
                """, (name, now, now))

    def get_files(self):
        with self.db:
            c = self.cursor()
            c.execute("""
                    SELECT id, name FROM file;
                """)

            ret = {}
            for index, data in enumerate(c.fetchall()):
                ret.update({index: {
                    "id": data[0],
                    "name": data[1],
                }})
            return ret

    def get_file_name(self, file_id):
        with self.db:
            c = self.cursor()
            c.execute("""
                    SELECT name FROM file WHERE id=?
                """, (file_id,))

            ret = c.fetchone()
            return ret[0]

    def remove_file(self, file_name):
        with self.db:
            c = self.cursor()
            c.execute("""
                    DELETE FROM file WHERE name=?
                """, (file_name,))

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
            d = topology._re_standardize(topology.number_data)
            input_data = cPickle.dumps(d)
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

# global storage
storage = Storage()
