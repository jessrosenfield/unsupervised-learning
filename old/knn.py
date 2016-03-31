from joblib import Parallel, delayed
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier

import data_util as util
import matplotlib.pyplot as plt
import numpy as np

PORTIONS = np.arange(.1, 1.1, .1)
K_NEIGHBORS = np.arange(1, 11)
bc_data_train, bc_data_test, bc_target_train, bc_target_test = util.load_breast_cancer()
v_data_train, v_data_test, v_target_train, v_target_test = util.load_vowel()

def knn_neighbors():
    print "knn_k_neighbors"
    print "---bc---"
    Parallel(n_jobs=-1)(
        delayed(_knn_neighbors)(
            bc_data_train,
            bc_data_test,
            bc_target_train,
            bc_target_test,
            n_neighbors) for n_neighbors in K_NEIGHBORS)
    print "---v---"
    Parallel(n_jobs=-1)(
        delayed(_knn_neighbors)(
            v_data_train,
            v_data_test,
            v_target_train,
            v_target_test,
            n_neighbors) for n_neighbors in K_NEIGHBORS)


def _knn_neighbors(data, data_test, target, target_test, n_neighbors):
    knn = KNeighborsClassifier(weights='distance', n_neighbors=n_neighbors)
    train_score = np.mean(cross_validation.cross_val_score(knn, data, target, cv=10))
    knn.fit(data, target)
    test_score = knn.score(data_test, target_test)
    print n_neighbors, train_score, test_score


def knn_train_size():
    print "train_size"
    print "---bc---"
    Parallel(n_jobs=-1)(
        delayed(_knn_train_size)(
            bc_data_train,
            bc_data_test,
            bc_target_train,
            bc_target_test,
            3,
            train_size) for train_size in PORTIONS)
    print "---v---"
    Parallel(n_jobs=-1)(
        delayed(_knn_train_size)(
            v_data_train,
            v_data_test,
            v_target_train,
            v_target_test,
            1,
            train_size) for train_size in PORTIONS if train_size > .1)


def _knn_train_size(data, data_test, target, target_test, n_neighbors, train_size):
    knn = KNeighborsClassifier(weights='distance', n_neighbors=n_neighbors)
    if train_size < 1:
        X_train, _, y_train, _ = cross_validation.train_test_split(
            data, target, train_size=train_size, stratify=target)
    else:
        X_train, y_train = data, target
    train_score = np.mean(cross_validation.cross_val_score(knn, X_train, y_train, cv=10))
    knn.fit(X_train, y_train)
    test_score = knn.score(data_test, target_test)
    print train_size, train_score, test_score


if __name__ == "__main__":
    knn_neighbors()
    knn_train_size()

