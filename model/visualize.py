import click
import torch
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from model.cnnfeatures import train_test_set, FeatureExtractor


def plot_images(images, classes, labels=None, normalize=False):
    n_images = len(images)
    rows, cols = int(np.sqrt(n_images)), int(np.sqrt(n_images))

    fig = plt.figure(figsize=(10, 10))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        image = images[i]

        if normalize:
            image_min = image.min()
            image_max = image.max()
            image.clamp_(min=image_min, max=image_max)
            image.add_(-image_min).div_(image_max - image_min + 1e-5)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        if labels is not None:
            ax.set_title(classes[labels[i]])
        ax.axis('off')
    fig.show()


@click.command()
@click.option('--nimages', type=int, default=9)
def draw_images(nimages):
    X_tr, *_ = train_test_set()
    images, labels = zip(*[X_tr[i]for i in range(nimages)])
    plot_images(images, labels)


@click.command()
@click.option('--size', type=int, default=1000)
def draw_cnn_features(size):
    X_tr, _, y_tr, _ = train_test_set()
    # Use a small sample to train TSNe
    data, _ = torch.utils.data.random_split(X_tr, [size, len(X_tr) - size])
    model = make_pipeline(
        FeatureExtractor(),
        TSNE(n_components=2),
    )
    plt.figure(figsize=(8, 4))
    X_x, X_y = model.fit_transform(X_tr).T
    plt.scatter(X_x, X_y, c=y_tr)
    plt.show()
