import numpy as np

_VOWEL_PATH = 'datasets/vowel/vowel.train'
_VOWEL_TEST_PATH = 'datasets/vowel/vowel.test'


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
    return (train[0], train[1], test[0], test[1])


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

