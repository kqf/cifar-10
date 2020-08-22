import pytest
import torch

from model.cnnfeatures import tolabels, train_test_set


@pytest.fixture(scope="session")
def data(size=100):
    # Test the train test set method here
    X_tr, *_ = train_test_set()
    data, _ = torch.utils.data.random_split(X_tr, [size, len(X_tr) - size])
    return data


@pytest.fixture(scope="session")
def labels(data):
    return tolabels(data)
