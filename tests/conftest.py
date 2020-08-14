import pytest
import torchvision

PRETRAINED_SIZE = 224
PRETRAINED_MEANS = [0.485, 0.456, 0.406]
PRETRAINED_STDS = [0.229, 0.224, 0.225]


@pytest.fixture
def train():
    # See https://pytorch.org/docs/stable/torchvision/models.html

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(PRETRAINED_SIZE),
        torchvision.transforms.RandomRotation(5),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomCrop(PRETRAINED_SIZE, padding=10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(PRETRAINED_MEANS, PRETRAINED_STDS)
    ])

    train = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )
    return train


@pytest.fixture
def validation():
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(PRETRAINED_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(PRETRAINED_MEANS, PRETRAINED_STDS)
    ])

    test = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )
    return test
