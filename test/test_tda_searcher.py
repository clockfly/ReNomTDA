import pytest
import numpy as np
from renom_tda import NumberSearcher, TextSearcher, SearcherResolver


def test_number_searcher_equal():
    numbers = np.array([[0., 1.], [2., 3.]])
    searcher = NumberSearcher(data=numbers, columns=["col1", "col2"])
    index = searcher.search(0, "=", 0)

    assert index == [0]


def test_number_searcher_gt():
    numbers = np.array([[0., 1.], [2., 3.]])
    searcher = NumberSearcher(data=numbers, columns=["col1", "col2"])
    index = searcher.search(0, ">", 0)

    assert index == [0, 1]


def test_number_searcher_st():
    numbers = np.array([[0., 1.], [2., 3.]])
    searcher = NumberSearcher(data=numbers, columns=["col1", "col2"])
    index = searcher.search(0, "<", 0)

    assert index == [0]


def test_number_searcher_none_operator():
    numbers = np.array([[0., 1.], [2., 3.]])
    searcher = NumberSearcher(data=numbers, columns=["col1", "col2"])
    index = searcher.search(0, None, 0)
    assert index == []


def test_number_searcher_none_column():
    numbers = np.array([[0., 1.], [2., 3.]])
    searcher = NumberSearcher(data=numbers, columns=["col1", "col2"])
    with pytest.raises(Exception):
        searcher.search(None, "<", 0)


def test_number_searcher_none_value():
    numbers = np.array([[0., 1.], [2., 3.]])
    searcher = NumberSearcher(data=numbers, columns=["col1", "col2"])
    with pytest.raises(Exception):
        searcher.search(0, "<", None)


def test_text_searcher_equal():
    texts = np.array([["a", "b"]])
    searcher = TextSearcher(data=texts, columns=["col1"])
    index = searcher.search(0, "=", "a")

    assert index == [0]


def test_text_searcher_like():
    texts = np.array([["abc", "def"]])
    searcher = TextSearcher(data=texts, columns=["col1"])
    index = searcher.search(0, "like", "b")

    assert index == [0]


def test_text_searcher_none_operator():
    texts = np.array([["a", "b"]])
    searcher = TextSearcher(data=texts, columns=["col1"])
    index = searcher.search(0, None, "a")
    assert index == []


def test_text_searcher_none_column():
    texts = np.array([["a", "b"]])
    searcher = TextSearcher(data=texts, columns=["col1"])
    with pytest.raises(Exception):
        searcher.search(None, "=", "a")


def test_text_searcher_none_value():
    texts = np.array([["a", "b"]])
    searcher = TextSearcher(data=texts, columns=["col1"])
    with pytest.raises(Exception):
        searcher.search(0, "=", None)


def test_resolver_number():
    numbers = np.array([[0., 1.], [2., 3.]])
    texts = np.array([["a", "b"]])
    resolver = SearcherResolver(numbers, texts, ["col1", "col2"], ["col3"])
    searcher = resolver.resolve("number")
    assert isinstance(searcher, NumberSearcher)


def test_resolver_text():
    numbers = np.array([[0., 1.], [2., 3.]])
    texts = np.array([["a", "b"]])
    resolver = SearcherResolver(numbers, texts, ["col1", "col2"], ["col3"])
    searcher = resolver.resolve("text")
    assert isinstance(searcher, TextSearcher)
