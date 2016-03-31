from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.externals.six import StringIO
from sklearn.feature_selection import RFECV

import data_util as util
import numpy as np
import pydot
import sklearn.tree as tree

bc_data_train, bc_data_test, bc_target_train, bc_target_test = util.load_breast_cancer()
bc_feature_names = []
v_data_train, v_data_test, v_target_train, v_target_test = util.load_vowel()
DEPTHS = np.arange(1, 10)

def decision_tree():
    print "---bc---"
    clf = tree.DecisionTreeClassifier(criterion="gini")

    rfecv = RFECV(clf, cv=10)

    _decision_tree(clf, bc_data_train, bc_data_test, bc_target_train, bc_target_test, "bc_gini")
    for depth in DEPTHS:
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=depth)
        _decision_tree(clf, bc_data_train, bc_data_test, bc_target_train, bc_target_test, "bc_gini" + str(depth))

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    _decision_tree(clf, bc_data_train, bc_data_test, bc_target_train, bc_target_test, "bc_entropy")
    for depth in DEPTHS:
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        _decision_tree(clf, bc_data_train, bc_data_test, bc_target_train, bc_target_test, "bc_entropy" + str(depth))

    rfecv.fit(bc_data_train, bc_target_train)
    print rfecv.support_
    print rfecv.ranking_
    print rfecv.score(bc_data_test, bc_target_test)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


    print "---v---"
    clf = tree.DecisionTreeClassifier(criterion="gini")

    rfecv = RFECV(clf, cv=10)

    _decision_tree(clf, v_data_train, v_data_test, v_target_train, v_target_test, "v_gini")
    for depth in DEPTHS:
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=depth)
        _decision_tree(clf, v_data_train, v_data_test, v_target_train, v_target_test, "v_gini" + str(depth))

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    _decision_tree(clf, v_data_train, v_data_test, v_target_train, v_target_test, "v_entropy")
    for depth in DEPTHS:
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        _decision_tree(clf, v_data_train, v_data_test, v_target_train, v_target_test, "v_entropy" + str(depth))

    rfecv.fit(v_data_train, v_target_train)
    print rfecv.support_
    print rfecv.ranking_
    print rfecv.score(v_data_test, v_target_test)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()



def _decision_tree(clf, data, data_test, target, target_test, name):
    clf.fit(data, target)
    train_score = clf.score(data, target)
    test_score = clf.score(data_test, target_test)
    print name, train_score, test_score
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    with open("output/dt-pics/" + name + ".png", 'w') as f:
        f.write(graph.create_png())


if __name__ == "__main__":
    decision_tree()
