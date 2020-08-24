import pytest

from model.cnnfeatures import build_model
from model.shallow import build_model as build_shallow_model
from model.deep import build_model as build_deep


@pytest.mark.parametrize("build", [
    build_model,
    build_shallow_model,
    build_deep
])
def test_handles_model(build, data, labels):
    model = build().fit(data, labels)
    assert model.score(data, labels) > 0.1
