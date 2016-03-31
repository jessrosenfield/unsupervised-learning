from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

import data_util as util
import numpy as np

bc_data_train, bc_data_test, bc_target_train, bc_target_test = util.load_breast_cancer()
v_data_train, v_data_test, v_target_train, v_target_test = util.load_vowel()
ESTIMATORS = np.arange(1, 101, 1)


def boosting():
    print "---bc---"
    for n_estimators in ESTIMATORS:
        _boosting(bc_data_train, bc_data_test, bc_target_train, bc_target_test, n_estimators)
    print "---v---"
    for n_estimators in ESTIMATORS:
        _boosting(v_data_train, v_data_test, v_target_train, v_target_test, n_estimators)


def _boosting(data, data_test, target, target_test, n_estimators):
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    clf.fit(data, target)
    train_score = clf.score(data, target)
    test_score = clf.score(data_test, target_test)
    print n_estimators, train_score, test_score


if __name__ == "__main__":
    boosting()



