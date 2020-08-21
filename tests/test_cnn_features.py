from model.cnnfeatures import FeatureExtractor


def test_creates_features(data):
    model = FeatureExtractor()
    assert model.transform(data).shape == (len(data), 512)
