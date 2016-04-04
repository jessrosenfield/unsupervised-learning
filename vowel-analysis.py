# source: https://github.com/joshuamorton/Machine-Learning/blob/master/P3/analysis.py
# source: https://github.com/iRapha/CS4641/blob/master/P3/analysis.py
# source: https://github.com/baiyishr/baiyishr.github.io/blob/master/MLprojects/unsupervisedlearning/tools.py

import argparse

from pprint import pprint
from StringIO import StringIO

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans as KM
from sklearn.decomposition import FastICA as ICA
from sklearn.decomposition.pca import PCA as PCA
from sklearn.feature_selection import SelectKBest as best
from sklearn.feature_selection import f_classif
from sklearn.mixture import GMM as EM
from sklearn.random_projection import GaussianRandomProjection as RandomProjection

from sknn.mlp import Classifier, Layer

import data_util as util



def plot(axes, values, x_label, y_label, title, name):
    print "plot" + title + name
    plt.clf()
    plt.plot(*values)
    plt.axis(axes)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig("plots/v/"+name+".png", dpi=500)
    # plt.show()
    plt.clf()


def pca(tx, ty, rx, ry):
    print "pca"
    compressor = PCA(n_components = tx[1].size/2)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wPCAtr")
    km(newtx, ty, newrx, ry, add="wPCAtr")
    nn(newtx, ty, newrx, ry, add="wPCAtr")
    print "pca done"


def ica(tx, ty, rx, ry):
    print "ica"
    compressor = ICA(whiten=True)  # for some people, whiten needs to be off
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wICAtr")
    km(newtx, ty, newrx, ry, add="wICAtr")
    nn(newtx, ty, newrx, ry, add="wICAtr")
    print "ica done"


def randproj(tx, ty, rx, ry):
    print "randproj"
    compressor = RandomProjection(tx[1].size)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    # compressor = RandomProjection(tx[1].size)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wRPtr")
    km(newtx, ty, newrx, ry, add="wRPtr")
    nn(newtx, ty, newrx, ry, add="wRPtr")
    print "randproj done"


def kbest(tx, ty, rx, ry):
    print "kbest"
    for i in range(9):
        k = i + 1
        add = "wKBtr" + str(k)
        compressor = best(f_classif, k=k)
        compressor.fit(tx, y=ty)
        newtx = compressor.transform(tx)
        newrx = compressor.transform(rx)
        em(newtx, ty, newrx, ry, add=add)
        km(newtx, ty, newrx, ry, add=add)
        nn(newtx, ty, newrx, ry, add=add)
    print "kbest done"


def em(tx, ty, rx, ry, add="", times=10):
    print "em" + add
    errs = []

    # this is what we will compare to
    checker = EM(n_components=2)
    checker.fit(rx)
    truth = checker.predict(rx)

    # so we do this a bunch of times
    for i in range(2,times):
        clusters = {x:[] for x in range(i)}

        # create a clusterer
        clf = EM(n_components=i)
        clf.fit(tx)  #fit it to our data
        test = clf.predict(tx)
        result = clf.predict(rx)  # and test it on the testing set

        # here we make the arguably awful assumption that for a given cluster,
        # all values in tha cluster "should" in a perfect world, belong in one
        # class or the other, meaning that say, cluster "3" should really be
        # all 0s in our truth, or all 1s there
        #
        # So clusters is a dict of lists, where each list contains all items
        # in a single cluster
        for index, val in enumerate(result):
            clusters[val].append(index)

        # then we take each cluster, find the sum of that clusters counterparts
        # in our "truth" and round that to find out if that cluster should be
        # a 1 or a 0
        mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}

        # the processed list holds the results of this, so if cluster 3 was
        # found to be of value 1,
        # for each value in clusters[3], processed[value] == 1 would hold
        processed = [mapper[val] for val in result]
        errs.append(sum((processed-truth)**2) / float(len(ry)))
    plot([0, times, min(errs)-.1, max(errs)+.1],[range(2, times), errs, "ro"], "Number of Clusters", "Error Rate", "Expectation Maximization Error", "EM"+add)

    # dank magic, wrap an array cuz reasons
    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    newtx = np.append(tx, td, 1)
    newrx = np.append(rx, rd, 1)
    nn(newtx, ty, newrx, ry, add="onEM"+add)
    print "em done" + add



def km(tx, ty, rx, ry, add="", times=10):
    print "km"
    #this does the exact same thing as the above
    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 50, 88] # eight for num speakers, eleven for num vowels
    orig = add
    for num_c in clusters:
        add = orig + "nc" + str(num_c)
        errs = []
        checker = KM(n_clusters=num_c)
        checker.fit(ry)
        truth = checker.predict(ry)

        # so we do this a bunch of times
        for i in range(2,times):
            clusters = {x:[] for x in range(i)}
            clf = KM(n_clusters=i)
            clf.fit(tx)  #fit it to our data
            test = clf.predict(tx)
            result = clf.predict(rx)  # and test it on the testing set
            for index, val in enumerate(result):
                clusters[val].append(index)
            mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}
            processed = [mapper[val] for val in result]
            errs.append(sum((processed-truth)**2) / float(len(ry)))
        plot([0, times, min(errs)-.1, max(errs)+.1],[range(2, times), errs, "ro"], "Number of Clusters", "Error Rate", "KMeans clustering error", "KM"+add)

        td = np.reshape(test, (test.size, 1))
        rd = np.reshape(result, (result.size, 1))
        newtx = np.append(tx, td, 1)
        newrx = np.append(rx, rd, 1)
        nn(newtx, ty, newrx, ry, add="onKM"+add)
    print "km done" + add


def nn(tx, ty, rx, ry, add="", iterations=4001):
    """
    trains and plots a neural network on the data we have
    """
    print "nn" + add
    resultst = []
    resultsr = []
    iter_arr = np.arange(iterations, step=500)
    iter_arr[0] = 1
    # queue = mp.Queue()
    # processes = []
    # processes = [mp.Process(target=_nn, args=[tx, ty, rx, ry, i_num]) for i_num in iter_arr]
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()
    # results = []
    # for _ in processes:
    #     results.append(queue.get());
    # for result in sorted(results, key=lambda x: x[0]):
    #     print result
    #     i_num, train_score, test_score = result
    #     resultst.append(train_score)
    #     resultsr.append(test_score)
    for i_num in iter_arr:
        result = _nn(tx, ty, rx, ry, i_num)
        print result
        resultst.append(1. - result[1])
        resultsr.append(1. - result[2])
    plot([0, iterations, 0, 1], (iter_arr, resultst, "ro", iter_arr, resultsr, "bo"), "Network Epoch", "Percent Error", "Neural Network Error", "NN"+add)
    print "nn done" + add

def _nn(tx, ty, rx, ry, n_iter):
    print "_nn"
    nn = Classifier(
            layers=[
                Layer("Tanh", units=100),
                Layer("Softmax")],
            n_iter=n_iter)
    nn.fit(tx, ty)
    resultst = nn.score(tx, ty)
    resultsr = nn.score(rx, ry)
    print "_nn done"
    return n_iter, resultst, resultsr



if __name__=="__main__":
    train_x, train_y, test_x, test_y = util.load_vowel()
    # em(train_x, train_y, test_x, test_y)
    # km(train_x, train_y, test_x, test_y)
    # pca(train_x, train_y, test_x, test_y)
    # ica(train_x, train_y, test_x, test_y)
    # randproj(train_x, train_y, test_x, test_y)
    kbest(train_x, train_y, test_x, test_y)
    # nn(train_x, train_y, test_x, test_y)
