from eval.metrics import exact_match, percentile, quality_report, rouge_l


def test_exact_match_normalises():
    assert exact_match(["The Answer."], ["the answer"]) == 1.0
    assert exact_match(["nope"], ["yes"]) == 0.0


def test_rouge_l_perfect_and_partial():
    assert rouge_l(["a b c d"], ["a b c d"]) == 1.0
    assert 0.0 < rouge_l(["a b x y"], ["a b c d"]) < 1.0


def test_percentile_monotonic():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(vals, 0.5) <= percentile(vals, 0.95)
    assert percentile([], 0.5) == 0.0


def test_quality_report_keys():
    rep = quality_report(["hello world"], ["hello world"])
    assert set(rep) == {"rougeL_f1", "exact_match", "n"}
    assert rep["n"] == 1.0
