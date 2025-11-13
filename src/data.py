import torch
from torchvision import datasets, transforms


class BarlowTwinsTransform:
    """
    A custom transform class for Barlow Twins. It applies a series of
    random augmentations to an image twice, returning two distorted views.
    """

    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
                ),
            ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return y1, y2


def get_cifar10_dataloaders(batch_size, for_evaluation=False):
    if for_evaluation:
        # Simple transform for the downstream linear evaluation
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
                ),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        # Barlow Twins transform for self-supervised pre-training
        transform = BarlowTwinsTransform()
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        # We don't need a test set for pre-training
        test_dataset = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    if test_dataset:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return train_loader, test_loader

    return train_loader, None
