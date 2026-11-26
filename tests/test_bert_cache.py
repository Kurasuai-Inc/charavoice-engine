import torch

from charavoice_engine.nlp.bert_cache import BertFeatureCache


def test_bert_cache_distinguishes_same_text_with_different_word2ph() -> None:
    cache = BertFeatureCache(max_size=10, device="cpu")
    feature_a = torch.ones((1024, 5))
    feature_b = torch.full((1024, 9), 2.0)

    cache.put("SQL", "EN", feature_a, word2ph=[1, 1, 1, 1, 1])
    cache.put("SQL", "EN", feature_b, word2ph=[1, 3, 2, 2, 1])

    cached_a = cache.get("SQL", "EN", word2ph=[1, 1, 1, 1, 1])
    cached_b = cache.get("SQL", "EN", word2ph=[1, 3, 2, 2, 1])

    assert cached_a is not None
    assert cached_b is not None
    assert cached_a.shape == (1024, 5)
    assert cached_b.shape == (1024, 9)
    assert torch.equal(cached_a, feature_a)
    assert torch.equal(cached_b, feature_b)
