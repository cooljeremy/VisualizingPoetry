from poetic_vis.data.datasets import PoeticVisDataset


def test_dataset_empty(tmp_path):
    d = PoeticVisDataset(str(tmp_path), "missing.jsonl")
    assert len(d) == 0


