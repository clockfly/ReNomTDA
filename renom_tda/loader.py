# -*- coding: utf-8 -*.t-
from __future__ import print_function
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass
import numpy as np
import pandas as pd


class Loader(with_metaclass(ABCMeta, object)):
    """Abstract class of data loading modules."""

    @abstractmethod
    def load(self):
        pass


class CSVLoader(Loader):
    """Loader of CSV file."""

    def __init__(self, file_name):
        self.file_name = file_name

    def load(self):
        file_data = pd.read_csv(self.file_name).dropna()
        text_index = (file_data.dtypes == "object")
        number_index = np.logical_or(file_data.dtypes == "float", file_data.dtypes == "int")

        numbers = file_data.loc[:, number_index]
        number_columns = numbers.columns
        texts = file_data.loc[:, text_index]
        text_columns = texts.columns

        return np.array(numbers), np.array(texts), number_columns, text_columns
