# -*- coding: utf-8 -*.t-
from __future__ import print_function
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass
import numpy as np


class Searcher(with_metaclass(ABCMeta, object)):
    def is_type(self, data_type):
        if data_type == self.data_type:
            return True
        return False

    @abstractmethod
    def search(self):
        pass


class NumberSearcher(Searcher):
    def __init__(self, data, columns):
        self.data_type = "number"
        self.data = np.array(data)
        self.columns = columns

    def search(self, column, operator, value):
        index = []
        if operator == "=":
            index.extend(np.where(self.data[:, column] == value)[0])
        elif operator == ">":
            index.extend(np.where(self.data[:, column] >= value)[0])
        if operator == "<":
            index.extend(np.where(self.data[:, column] <= value)[0])
        return index


class TextSearcher(Searcher):
    def __init__(self, data, columns):
        self.data_type = "text"
        self.data = np.array(data)
        self.columns = columns

    def search(self, column, operator, value):
        index = []
        if operator == "=":
            index.extend(np.where(self.data == value)[0])
        elif operator == "like":
            for i, d in enumerate(self.data):
                if value in d:
                    index.append(i)
        return index


class SearcherResolver(object):
    def __init__(self, numbers, texts, number_columns, text_columns):
        self.searcher_list = [
            NumberSearcher(numbers, number_columns),
            TextSearcher(texts, text_columns)]

    def resolve(self, data_type):
        for s in self.searcher_list:
            if s.is_type(data_type):
                return s
