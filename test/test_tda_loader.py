import pytest
from renom_tda.loader import CSVLoader


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
