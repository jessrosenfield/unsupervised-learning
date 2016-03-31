from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
import numpy as np

_breast_cancer_path = 'datasets/wisconsin/breast-cancer-wisconsin.data'
_vowel_path = 'datasets/vowel/vowel.train'
_vowel_test_path = 'datasets/vowel/vowel.test'


def _is_nan(string):
    if string is not '?':
        return int(string)
    else:
        return np.nan


def _initialize_breast_cancer():
    cnv = {6: lambda s: _is_nan(s)}
    bc_data = np.loadtxt(_breast_cancer_path, delimiter=',', converters=cnv)
    bc_data_malignant = bc_data[bc_data[:, 10] == 4, :]
    bc_data_benign = bc_data[bc_data[:, 10] == 2, :]

    imp = Imputer(missing_values=np.nan, strategy='median', axis=0)

    bc_data = np.concatenate((imp.fit_transform(bc_data_malignant),
            imp.fit_transform(bc_data_benign))).astype(int)
    X = bc_data[:, 1:10]
    y = bc_data[:, 10]
    bc_data_train, bc_data_test, bc_target_train, bc_target_test = train_test_split(X, y, test_size=.3, stratify=y)
    bc_data_train.dump('bc_data.train')
    bc_data_test.dump('bc_data.test')
    bc_target_train.dump('bc_target.train')
    bc_target_test.dump('bc_target.test')


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
    bc_data_train = np.load('bc_data.train')
    bc_data_test = np.load('bc_data.test')
    bc_target_train = np.load('bc_target.train')
    bc_target_test = np.load('bc_target.test')
    return (bc_data_train, bc_data_test, bc_target_train, bc_target_test)


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
    return (train[0], test[0], train[1], test[1])


def _load_vowel_train():
    vowel_data = np.loadtxt(_vowel_path, delimiter=',', skiprows=1)
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
    vowel_data = np.loadtxt(_vowel_test_path, delimiter=',', skiprows=1)
    X = vowel_data[:, -10:]
    y = vowel_data[:, 1].astype(int)
    return (X, y)

