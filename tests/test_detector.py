from repcs.detector import kl_score, is_contaminated

def test_kl_positive():
    sparse = [('d1', 1.0), ('d2', 0.5)]
    dense  = [('d1', 0.9), ('d2', 0.4)]
    assert kl_score(sparse, dense) >= 0
