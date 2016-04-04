import numpy as np

_VOWEL_PATH = 'datasets/vowel/vowel.train'
_VOWEL_TEST_PATH = 'datasets/vowel/vowel.test'
_BREAST_CANCER_FOLDER = 'datasets/wisconsin/'

def load_vowel():
    """Load and return the vowel training dataset.

    Returns
    -------
    (X_train, X_test, y_train, y_test) Tuple
        A tuple of data and target

    The copy of the vowel dataset is downloaded from:
    http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
    """
    train = _load_vowel_train()
    test = _load_vowel_test()
    return (train[0], train[1].reshape(-1, 1), test[0], test[1].reshape(-1, 1))


def _load_vowel_train():
    vowel_data = np.loadtxt(_VOWEL_PATH, delimiter=',', skiprows=1)
    X = vowel_data[:, -10:]
    y = vowel_data[:, 1].astype(int)
    return (X, y)


def _load_vowel_test():
    """Load and return the vowel testing dataset.

    Returns
    -------
    (X, y) Tuple
        A tuple of data and target

    The copy of the vowel dataset is downloaded from:
    http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
    """
    vowel_data = np.loadtxt(_VOWEL_TEST_PATH, delimiter=',', skiprows=1)
    X = vowel_data[:, -10:]
    y = vowel_data[:, 1].astype(int)
    return (X, y)

def load_breast_cancer():
    """Load and return the breast cancer wisconsin dataset (classification).
    The breast cancer dataset is a classic and very easy binary classification
    dataset.

    Returns
    -------
    (X_train, X_test, y_train, y_test) Tuple
        A tuple of data and target

    The copy of UCI ML Breast Cancer Wisconsin (Original) dataset is
    downloaded from:
    http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
    """
    bc_data_train = np.load(_BREAST_CANCER_FOLDER+'bc_data.train')
    bc_data_test = np.load(_BREAST_CANCER_FOLDER+'bc_data.test')
    bc_target_train = np.load(_BREAST_CANCER_FOLDER+'bc_target.train')
    bc_target_test = np.load(_BREAST_CANCER_FOLDER+'bc_target.test')
    for i in range(len(bc_target_test)):
        if bc_target_test[i] == 2:
            bc_target_test[i] = 0
        elif bc_target_test[i] == 4:
            bc_target_test[i] = 1
    for i in range(len(bc_target_train)):
        if bc_target_train[i] == 2:
            bc_target_train[i] = 0
        elif bc_target_train[i] == 4:
            bc_target_train[i] = 1
    return (bc_data_train, bc_target_train.reshape(-1, 1), bc_data_test, bc_target_test.reshape(-1, 1))

