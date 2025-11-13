import argparse
import yaml
from poetic_vis.inference.generate import run_inference
from poetic_vis.inference.save_io import save_results


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()
    _ = load_yaml(args.config)
    outs = run_inference([args.input])
    path = save_results(outs, args.out_dir)
    print(path)


if __name__ == "__main__":
    main()


