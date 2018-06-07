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
    """Loader of CSV file.

    Params:
        file_name: load file name.
    """

    def __init__(self, file_name):
        self.file_name = file_name

    def load(self):
        """Function of loading data.

        Return:
            number_data: numpy array of number data.

            text_data: numpy array of text data.

            number_columns: array of number data column names.

            text_columns: array of text data column names.
        """
        file_data = pd.read_csv(self.file_name).dropna()
        text_index = (file_data.dtypes == "object")
        number_index = np.logical_or(file_data.dtypes == "float", file_data.dtypes == "int")

        numbers = file_data.loc[:, number_index]
        number_columns = numbers.columns
        texts = file_data.loc[:, text_index]
        text_columns = texts.columns

        return np.array(numbers), np.array(texts), number_columns, text_columns
