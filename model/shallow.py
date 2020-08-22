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


def tolabels(data):
    return [l for _, l in data]


def train_test_set():
    # Read the data without any transformations (to calculate stats)
    raw = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
    )

    means = raw.data.mean(axis=(0, 1, 2)) / 255
    stds = raw.data.std(axis=(0, 1, 2)) / 255

    # Normalize the features
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(means, stds)
    ])

    X_tr = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    y_tr = tolabels(X_tr)

    X_te = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    y_te = tolabels(X_te)
    return X_tr, X_te, y_tr, y_te


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
        SVC(C=0.1),
    )
    return model


def main():
    X_tr, X_te, y_tr, y_te = train_test_set()
    return
    model = build_model()
    model.fit(X_tr, y_tr)

    f1_tr = f1_score(y_tr, model.predict(X_tr), average="macro")
    print(f"Train f1: {f1_tr: <5}")

    f1_te = f1_score(y_te, model.predict(X_te), average="macro")
    print(f"Test f1: {f1_te: <5}".format())


if __name__ == "__main__":
    main()
