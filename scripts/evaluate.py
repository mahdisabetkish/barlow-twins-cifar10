import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm

from src.data import get_cifar10_dataloaders
from src.model import LinearClassifier


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = resnet18()
    backbone.fc = nn.Identity()  # Remove the original classifier
    backbone.load_state_dict(torch.load(args.checkpoint_path))
    backbone = backbone.to(device)

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    classifier = LinearClassifier(num_features=512, num_classes=10).to(device)

    train_loader, test_loader = get_cifar10_dataloaders(
        args.batch_size, for_evaluation=True
    )

    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("--- Training Linear Classifier ---")
    for epoch in range(args.epochs):
        classifier.train()
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            features = backbone(images)

            outputs = classifier(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print("--- Evaluating Classifier ---")
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on CIFAR-10 test set: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Evaluation on CIFAR-10")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/barlow_twins_backbone.pth",
        help="Path to the pre-trained backbone checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs for the classifier",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the classifier",
    )
    args = parser.parse_args()
    main(args)
