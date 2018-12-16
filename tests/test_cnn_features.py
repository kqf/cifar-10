import pytest
from model.model import CNNFeatures
from model.data import train_test_sample


@pytest.mark.filterwarnings("ignore:RuntimeWarning")
def test_creates_features(datapath):
    X_tr, X_te, y_tr, y_te = train_test_sample(datapath, size=0.1)

    print("Total training size", X_tr.shape[0], "images")
    print("Total training size", X_tr.nbytes / 1024 / 1024, "mb")

    features = CNNFeatures().fit_transform(X_tr)
    assert features.shape == (X_tr.shape[0], 1000)
