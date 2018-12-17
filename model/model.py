import tensorflow as tf
import tensornets
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA


class ReportShape(BaseEstimator, TransformerMixin):
    def __init__(self, msg):
        self.msg = msg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Reporting shape for {}: {}".format(self.msg, X.shape))
        return X


class CNNFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, net=tensornets.ResNet50, stem=False, n_batches=10):
        self.net = net
        self.model = None
        self.inputs = None
        self.stem = stem
        self.n_batches = n_batches

    def fit(self, X, y=None):
        input_shape = list(X.shape)
        input_shape[0] = None
        self.inputs = tf.placeholder(tf.float32, shape=input_shape)
        self.model = self.net(self.inputs, stem=self.stem)
        return self

    def transform(self, X):
        size = X.shape[0]
        _, n_features = self.model.get_shape().as_list()
        preds = np.zeros((size, n_features))
        with tf.Session() as sess:
            for indeces in np.array_split(np.arange(size), self.n_batches):
                imgs = self.model.preprocess(X[indeces].astype(np.float32))
                sess.run(self.model.pretrained())
                preds[indeces] = sess.run(self.model, {self.inputs: imgs})
        return preds


def build_model():
    model = make_pipeline(
        CNNFeatures(),
        ReportShape("CNN features"),
        PCA(n_components=1000, whiten=True),
        SVC(gamma="scale"),
    )
    return model


def build_shallow_model():
    model = make_pipeline(
        FunctionTransformer(
            lambda x: x.reshape(x.shape[0], -1),
            validate=False),
        ReportShape("Pixel features"),
        PCA(n_components=100, whiten=True),
        ReportShape("PCA"),
        SVC(gamma="scale"),
    )
    # This yields following result
    # Train score 0.8108
    # Test  score 0.4665
    # It's clear overfit, one can regularize this model
    # by decreasing C, or by reducing n_components
    return model
