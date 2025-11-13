def to_markdown(metrics: dict) -> str:
    keys = list(metrics.keys())
    header = "|" + "|".join(keys) + "|"
    sep = "|" + "|".join(["---"] * len(keys)) + "|"
    vals = "|" + "|".join([f"{metrics[k]:.4f}" if isinstance(metrics[k], float) else str(metrics[k]) for k in keys]) + "|"
    return "\n".join([header, sep, vals])


