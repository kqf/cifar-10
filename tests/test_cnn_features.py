from model.model import build_features


def test_creates_features(data):
    model = build_features()
    model.initialize()
    assert model.transform(data).shape == (len(data), 512)
