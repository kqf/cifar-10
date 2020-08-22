import skorch
import torch
import torchvision

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score

"""
# colab install

!pip install skorch torch torchvision scikit-learn
"""


def tolabels(data):
    return [l for _, l in data]


def train_test_set():
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

    X_te = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    y_te = tolabels(X_te)


class VisualModule(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # Remove the last layer
        self.backbone.fc = torch.nn.Identity()

        # Freze all the parameters
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, device=torch.device("cpu"), batch_size=128):
        self.batch_size = batch_size
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        self.backbone.to(device)
        self.device = device

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = torch.utils.data.DataLoader(X, batch_size=self.batch_size)

        output = []
        for batch, y in data:
            with torch.no_grad():
                output.append(self.backbone(batch.to(self.device)))

        return torch.cat(output).detach().cpu().numpy()


class ReportShape(BaseEstimator, TransformerMixin):
    def __init__(self, msg):
        self.msg = msg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Reporting shape for {}: {}".format(self.msg, X.shape))
        return X


def build_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = make_pipeline(
        FeatureExtractor(device),
        ReportShape("CNN features"),
        SVC(C=2),
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
