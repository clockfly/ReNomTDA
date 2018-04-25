import pytest
import numpy as np
from numpy.testing import assert_array_equal
from renom_tda.loader import CSVLoader


def test_csv_loader():
    file_name = "data/test.csv"
    loader = CSVLoader(file_name=file_name)
    numbers, texts, number_columns, text_columns = loader.load()

    test_numbers = np.array([[1.0, 2.0], [3.0, 4.0]])
    test_texts = np.array([["a"], ["b"]])

    assert_array_equal(numbers, test_numbers)
    assert_array_equal(texts, test_texts)


def test_csv_loader_file_not_exists():
    file_name = ""
    loader = CSVLoader(file_name=file_name)

    with pytest.raises(Exception):
        loader.load()


def test_csv_loader_file_name_is_none():
    file_name = None
    loader = CSVLoader(file_name=file_name)

    with pytest.raises(Exception):
        loader.load()
