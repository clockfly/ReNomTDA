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


    def register_file(self, name):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                    INSERT INTO file(name, created, updated)
                    VALUES (?, ?, ?)
                """, (name, now, now))

    def remove_file(self, file_name):
        with self.db:
            c = self.cursor()
            c.execute("""
                    DELETE FROM file WHERE name=?
                """, (file_name,))

    def exist_file(self, file_name):
        with self.db:
            c = self.cursor()
            c.execute("""
                    SELECT count(*) FROM file WHERE name=?
                """, (file_name,))
            for data in c:
                if data[0] > 0:
                    return True
                else:
                    return False

    def fetch_file(self, file_id):
        with self.db:
            c = self.cursor()
            c.execute("""
                    SELECT id, name FROM file WHERE id=?;
                """, (file_id,))

            ret = {}
            for file_id, file_name in c:
                ret = {
                    "id": file_id,
                    "name": file_name,
                }
            return ret

    def fetch_files(self):
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


# global storage
storage = Storage()
