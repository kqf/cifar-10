import torch
import torchvision

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score

"""
# colab install

!pip install torch torchvision scikit-learn
"""


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = []
        for i, (x, y) in enumerate(X):
            data.append(x.reshape(1, -1))
        return torch.cat(data).detach().cpu().numpy()


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
        FeatureExtractor(),
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
