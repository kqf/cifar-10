import skorch
import torch
import torchvision
import numpy as np


def tolabels(data):
    return np.array([l for _, l in data])


def train_test_set():
    # See https://pytorch.org/docs/stable/torchvision/models.html
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(pretrained_size),
        torchvision.transforms.RandomRotation(5),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomCrop(pretrained_size, padding=10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(pretrained_means, pretrained_stds)
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(pretrained_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(pretrained_means, pretrained_stds)
    ])

    train = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )
    return train, test, tolabels(train), tolabels(test)


class ClassifierModule(torch.nn.Module):
    def __init__(self, backbone, output_dim):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 224 * 224, output_dim)

    def forward(self, x):
        return self.fc(x.reshape(x.shape[0], -1))


def build_model(lr=1e-4, max_epochs=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = skorch.NeuralNetClassifier(
        module=ClassifierModule,
        module__backbone=None,
        module__output_dim=10,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=lr,
        max_epochs=max_epochs,
        batch_size=256,
        iterator_train__shuffle=True,
        iterator_valid__shuffle=False,
        device=device,
    )
    return model


def main():
    X_tr, X_te, y_tr, y_te = train_test_set()
    model = build_model().fit(X_tr, y_tr)
    print(model.score(X_tr, y_tr))
    print(model.score(X_te, y_te))


if __name__ == "__main__":
    main()
