import argparse
import os

import torch
import torch.optim as optim
from tqdm import tqdm

from src.data import get_cifar10_dataloaders
from src.loss import barlow_twins_loss
from src.model import BarlowTwins


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, _ = get_cifar10_dataloaders(args.batch_size)

    model = BarlowTwins().to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1.5e-6
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for (y1, y2), _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            y1, y2 = y1.to(device), y2.to(device)

            optimizer.zero_grad()

            z1, z2 = model(y1, y2)
            loss = barlow_twins_loss(z1, z2, lambda_param=args.lambda_param)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

    # Save the backbone (encoder)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.backbone.state_dict(), "checkpoints/barlow_twins_backbone.pth")
    print("Pre-trained backbone saved to checkpoints/barlow_twins_backbone.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Barlow Twins Pre-training on CIFAR-10"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--lambda_param", type=float, default=5e-3, help="Lambda for loss function"
    )
    args = parser.parse_args()
    main(args)
