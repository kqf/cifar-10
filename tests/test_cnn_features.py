from model.model import build_features


def test_creates_features(train, validation):
    print(len(train))
    model = build_features()
    model.initialize()
    model.predict(validation)
