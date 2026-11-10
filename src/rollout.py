"""Evaluation and rollout utilities."""

import json
from pathlib import Path

import torch


@torch.no_grad()
def rollout(model, loader, device="cuda"):
    model.eval()
    records = []
    for batch in loader:
        outputs = model(batch["inputs"].to(device))
        for name, pred in zip(batch["names"], outputs.cpu().tolist()):
            records.append({"name": name, "prediction": pred})
    return records


def save_rollouts(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/rollouts.json")
    args = parser.parse_args()
    print(f"loading {args.checkpoint}; writing rollouts to {args.output}")
