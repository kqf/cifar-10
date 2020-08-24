import skorch
import torch
import torchvision


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
    # See https://pytorch.org/docs/stable/torchvision/models.html
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    print(f'Calculated means: {pretrained_means}')
    print(f'Calculated stds: {pretrained_stds}')

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(device).fit(train)

    test = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )
    print(model.predict(test))


if __name__ == "__main__":
    main()
