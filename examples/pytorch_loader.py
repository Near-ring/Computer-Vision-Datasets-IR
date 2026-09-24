"""Run a real CPU forward/backward pass over a packed dataset."""

from __future__ import annotations

import argparse

import torch
from cv_dataset_ir.torch import PackedTorchDataset, collate_samples
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    dataset = PackedTorchDataset(args.path)
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=args.workers,
        collate_fn=collate_samples,
        multiprocessing_context="spawn" if args.workers else None,
    )
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 1),
    ).cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    count = 0
    for images, targets in loader:
        optimizer.zero_grad()
        # Variable-size crops are legal; a batch can also be padded before stacking.
        loss = torch.stack(
            [model(image.unsqueeze(0).float() / 255).square().mean() for image in images]
        ).mean()
        loss.backward()
        optimizer.step()
        count += len(targets)
        print(f"samples={count}, loss={loss.item():.6f}, device=cpu")
    dataset.close()


if __name__ == "__main__":
    main()
