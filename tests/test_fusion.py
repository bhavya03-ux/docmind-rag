import pytest

from docmind.fusion import reciprocal_rank_fusion as rrf


def test_item_in_both_lists_wins():
    assert rrf([["a", "b"], ["b", "c"]])[0][0] == "b"


def test_score_formula():
    assert rrf([["x"]], k=60) == [("x", 1 / 61)]


def test_duplicates_in_one_list_count_once():
    assert rrf([["x", "x", "x"]], k=0) == [("x", 1.0)]


def test_ties_are_deterministic():
    assert [i for i, _ in rrf([["b"], ["a"]])] == ["a", "b"]


def test_empty_input():
    assert rrf([]) == []


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        rrf([["a"]], k=-1)
