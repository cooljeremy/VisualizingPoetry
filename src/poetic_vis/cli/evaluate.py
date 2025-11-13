import argparse
import yaml
from poetic_vis.evaluation.evaluator import evaluate_all
from poetic_vis.evaluation.reporting import to_markdown


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    _ = load_yaml(args.config)
    metrics = evaluate_all()
    print(to_markdown(metrics))


if __name__ == "__main__":
    main()


