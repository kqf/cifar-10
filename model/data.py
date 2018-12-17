import numpy as np
import pandas as pd
import sys
import pickle
from sklearn.model_selection import train_test_split


def sample(X, y, size):
    X, _, y, _ = train_test_split(X, y, train_size=size)
    return X, y

def load_batch(folder_path):
    with open(folder_path, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape(
        (len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels


def train_test(datapath):
    batches = [load_batch(f"{datapath}/data_batch_{i}") for i in range(1, 6)]
    X_tr, y_tr = zip(*batches)
    X_te, y_te = load_batch(f"{datapath}/test_batch")
    return np.concatenate(X_tr), X_te, np.concatenate(y_tr), np.array(y_te)


def train_test_sample(datapath, size):
    X_tr, X_te, y_tr, y_te = train_test(datapath)
    X_tr, y_tr = sample(X_tr, y_tr, size=size)
    X_te, y_te = sample(X_te, y_te, size=size)
    return X_tr, X_te, y_tr, y_te
