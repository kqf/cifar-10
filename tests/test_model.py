import pytest
from model.model import build_model, build_shallow_model
from model.data import train_test_sample


@pytest.mark.skip()
@pytest.mark.parametrize("build", [
    build_model,
    build_shallow_model
])
def test_handles_model(datapath, build):
    X_tr, X_te, y_tr, y_te = train_test_sample(datapath, size=0.001)
    model = build(n_components=2).fit(X_tr, y_tr)
    assert model.score(X_tr, y_tr) > 0.1
