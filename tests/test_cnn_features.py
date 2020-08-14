from model.model import build_features


def test_creates_features(data):
    model = build_features()
    model.initialize()
    assert model.predict_proba(data).shape == (len(data), 1000)
