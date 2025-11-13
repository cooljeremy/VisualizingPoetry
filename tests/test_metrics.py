from poetic_vis.evaluation.evaluator import evaluate_all


def test_evaluate_all():
    m = evaluate_all()
    assert "is" in m


