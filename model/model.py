import skorch
import torch
import torchvision
from sklearn.metrics import accuracy_score


class VisualModule(torch.nn.Module):
    def __init__(self, backbone, classify=False, n_output=10):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class VisionClassifierNet(skorch.NeuralNet):
    def predict(self, dataset):
        probas = self.predict_proba(dataset)
        return probas.argmax(-1)

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(preds, y)


def build_features(max_epochs=2, lr=1e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone = torchvision.models.resnet18(pretrained=True)
    model = VisionClassifierNet(
        module=VisualModule,
        module__backbone=backbone,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=lr,
        max_epochs=max_epochs,
        batch_size=10,
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
    # print(model.predict(test))


if __name__ == "__main__":
    main()
