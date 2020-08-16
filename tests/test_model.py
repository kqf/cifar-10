import pytest

from model.model import build_model
# from model.model import build_shallow_model


@pytest.mark.parametrize("build", [
    build_model,
    # build_shallow_model
])
def test_handles_model(build, data):
    model = build().fit(data)
    assert model.score(data) > 0.1
