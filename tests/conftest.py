import pytest
import torch
import torchvision


@pytest.fixture
def data():
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(pretrained_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(pretrained_means, pretrained_stds)
    ])

    test = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )
    data, _ = torch.utils.data.random_split(test, [100, len(test) - 100])
    return data


@pytest.fixture
def labels(data):
    return [l for _, l in data]
