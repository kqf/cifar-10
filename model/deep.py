import skorch
import torch
import torchvision
import numpy as np
from sklearn.metrics import f1_score


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
    def __init__(self, backbone, output_dim, freeze=True):
        super().__init__()
        self.backbone = backbone

        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze

        in_features = backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, output_dim)

    def forward(self, x):
        return self.backbone(x)


class ReportParams(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        n_pars = self.count_parameters(net.module_)
        print(f"The model has {n_pars:,} trainable parameters")

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(lr=1e-4, max_epochs=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = skorch.NeuralNetClassifier(
        module=ClassifierModule,
        module__backbone=torchvision.models.resnet18(pretrained=True),
        module__output_dim=10,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__param_groups=[
            ('conv1.*', {'lr': lr / 10}),
            ('bn1.*', {'lr': lr / 10}),
            ('layer1.*', {'lr': lr / 8}),
            ('layer2.*', {'lr': lr / 6}),
            ('layer3.*', {'lr': lr / 4}),
            ('layer4.*', {'lr': lr / 2}),
        ],
        optimizer__lr=lr,
        max_epochs=max_epochs,
        batch_size=256,
        iterator_train__shuffle=True,
        iterator_valid__shuffle=False,
        device=device,
        callbacks=[
            ReportParams()
        ],
    )
    return model


def main():
    X_tr, X_te, y_tr, y_te = train_test_set()
    model = build_model()
    model.fit(X_tr, y_tr)

    f1_tr = f1_score(y_tr, model.predict(X_tr), average="macro")
    print(f"Train f1: {f1_tr: <5}")

    f1_te = f1_score(y_te, model.predict(X_te), average="macro")
    print(f"Test f1: {f1_te: <5}".format())


if __name__ == "__main__":
    main()
