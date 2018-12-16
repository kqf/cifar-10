import numpy as np
import pandas as pd
import sys
import pickle


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

    possible_indices = np.arange(0, X_tr.shape[0])
    indices = np.random.choice(possible_indices,
                               size=int(X_tr.shape[0] * size), replace=False)
    return X_tr[indices], X_te, y_tr[indices], y_te
