from docmind.bm25 import BM25Index, tokenize

DOCS = [
    "Docker packages applications into containers",
    "Git tracks changes with commits and branches",
    "BM25 ranks documents by keyword relevance",
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Quick, brown FOX!") == ["quick", "brown", "fox"]


def test_best_match_first():
    assert BM25Index(DOCS).search("git branches", k=3)[0][0] == 1


def test_unknown_terms_return_nothing():
    assert BM25Index(DOCS).search("zzzz qqqq") == []


def test_empty_corpus_and_empty_query():
    assert BM25Index([]).search("anything") == []
    assert BM25Index(DOCS).search("") == []


def test_rare_term_outweighs_common_term():
    idx = BM25Index(["common common common rare", "common common common common", "common word here"])
    assert idx.search("common rare")[0][0] == 0


def test_k_limits_results():
    assert len(BM25Index(DOCS).search("docker git bm25 keyword commits containers", k=2)) <= 2


def test_scores_sorted_descending():
    scores = [s for _, s in BM25Index(DOCS).search("docker git bm25 commits containers", k=3)]
    assert scores == sorted(scores, reverse=True)
