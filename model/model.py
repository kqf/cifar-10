import skorch
import torch
import torchvision

from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score

"""
# colab install

!pip install skorch torch torchvision scikit-learn
"""


class VisualModule(torch.nn.Module):
    def __init__(self, backbone, flat=False):
        super().__init__()
        self.backbone = backbone
        self.flat = flat

        # Remove the last layer
        self.backbone.fc = torch.nn.Identity()

        # Freze all the parameters
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        if self.flat:
            return x.reshape(x.shape[0], -1)

        return self.backbone(x)


class FlatModule(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class FeatureExtractorNet(skorch.NeuralNet):
    def predict(self, dataset):
        probas = self.predict_proba(dataset)
        return probas.argmax(-1)

    def fit(self, X, y=None):
        self.initialize()
        return self

    def transform(self, dataset):
        return self.predict_proba(dataset)

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(preds, y)


def build_features(flat=False, max_epochs=2, lr=1e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone = torchvision.models.resnet18(pretrained=True)
    model = FeatureExtractorNet(
        module=VisualModule,
        module__backbone=backbone,
        module__flat=flat,
        criterion=torch.nn.CrossEntropyLoss,  # Not used
        iterator_train=None,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=4,
        device=device,
    )
    return model


class ReportShape(BaseEstimator, TransformerMixin):
    def __init__(self, msg):
        self.msg = msg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Reporting shape for {}: {}".format(self.msg, X.shape))
        return X


def build_model():
    model = make_pipeline(
        build_features(),
        ReportShape("CNN features"),
        SVC(),
    )
    return model


def build_shallow_model():
    model = make_pipeline(
        build_features(flat=True),
        ReportShape("CNN features"),
        SVC(),
    )
    return model


def tolabels(data):
    return [l for _, l in data]


def main():
    # See https://pytorch.org/docs/stable/torchvision/models.html
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(pretrained_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(pretrained_means, pretrained_stds)
    ])

    X_tr = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    y_tr = tolabels(X_tr)

    model = build_model()
    model.fit(X_tr, y_tr)

    X_te = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    y_te = tolabels(X_te)

    f1_tr = f1_score(y_tr, model.predict(X_tr), average="macro")
    print(f"Train f1: {f1_tr: <5}")

    f1_te = f1_score(y_te, model.predict(X_te), average="macro")
    print(f"Test f1: {f1_te: <5}".format())


if __name__ == "__main__":
    main()
