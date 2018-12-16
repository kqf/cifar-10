import click
from sklearn.pipeline import make_pipeline
# from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from model.data import train_test, train_test_sample
from model.model import CNNFeatures, ReportShape, Resizer


@click.command()
@click.option('--datapath',
              type=click.Path(exists=True),
              help='Path to the CIFAR-10 dataset',
              required=True)
def draw_images(datapath, nimages=10):
    X_tr, _, y_tr, _ = train_test(datapath)
    classes, index = set(), 0
    fig = plt.figure(figsize=(8, 4))
    while len(classes) < 10:
        if y_tr[index] in classes:
            index += 1
            continue
        fig.add_subplot(2, nimages / 2, len(classes) + 1)
        plt.imshow(X_tr[index])
        classes.add(y_tr[index])
    plt.tight_layout()
    plt.show()


@click.command()
@click.option('--datapath',
              type=click.Path(exists=True),
              help='Path to the CIFAR-10 dataset',
              required=True)
def draw_cnn_features(datapath):
    X_tr, _, y_tr, _ = train_test_sample(datapath, size=0.1)
    model = make_pipeline(
        CNNFeatures(n_batches=1),
        ReportShape("CNN features"),
        TSNE(n_components=2),
    )
    plt.figure(figsize=(8, 4))
    X_x, X_y = model.fit_transform(X_tr).T
    plt.scatter(X_x, X_y, c=y_tr)
    plt.show()
