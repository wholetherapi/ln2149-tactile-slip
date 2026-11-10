"""Training entry point."""

import argparse

import torch

from src.dataset import build_loader
from src.model import build_model


def train(args):
    loader = build_loader(args.data_dir, args.batch_size)
    model = build_model(args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            loss = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")
    torch.save(model.state_dict(), args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="checkpoints/model.pt")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
