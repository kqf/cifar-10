import click
from model.model import build_model, build_shallow_model
from model.data import train_test_sample


@click.command()
@click.option('--datapath',
              type=click.Path(exists=True),
              help='Path to the CIFAR-10 dataset',
              required=True)
def train_shallow_model(datapath):
    X_tr, X_te, y_tr, y_te = train_test_sample(datapath, size=0.1)
    model = build_shallow_model().fit(X_tr, y_tr)
    # NB: Use accuracy as classes are ballanced
    print("Train score", model.score(X_tr, y_tr))
    print("Test  score", model.score(X_te, y_te))


@click.command()
@click.option('--datapath',
              type=click.Path(exists=True),
              help='Path to the CIFAR-10 dataset',
              required=True)
def train_model(datapath):
    X_tr, X_te, y_tr, y_te = train_test_sample(datapath, size=0.1)
    model = build_model().fit(X_tr, y_tr)

    print("Train score", model.score(X_tr, y_tr))
    print("Test  score", model.score(X_te, y_te))
