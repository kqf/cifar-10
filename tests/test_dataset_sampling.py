import pytest
from model.data import train_test_sample


@pytest.mark.skip()
@pytest.mark.parametrize("size", [0.5, 0.1, 0.2])
def test_creates_features(datapath, size):
    X_tr, X_te, y_tr, y_te = train_test_sample(datapath, size)
    assert X_tr.shape[0] == y_tr.shape[0]
