import data_util as util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time

from sklearn import metrics
from sklearn.cluster import KMeans as KM
from sklearn.decomposition import FastICA as ICA
from sklearn.decomposition.pca import PCA as PCA
from sklearn.feature_selection import SelectKBest as best
from sklearn.feature_selection import chi2, f_classif
from sklearn.mixture import GMM as EM
from sklearn.random_projection import GaussianRandomProjection as RandomProjection

def dimensional(tx, ty, rx, ry, add=None):
    print "pca"
    for j in range(tx[1].size):
        i = j + 1
        print "===" + str(i)
        compressor = PCA(n_components = i)
        t0 = time()
        compressor.fit(tx, y=ty)
        newtx = compressor.transform(tx)
        runtime=time() - t0
        V = compressor.components_
        print runtime, V.shape, compressor.score(tx)
        distances = np.linalg.norm(tx-compressor.inverse_transform(newtx))
        print distances
    print "pca done"
    print "ica"
    for j in range(tx[1].size):
        i = j + 1
        print "===" + str(i)
        compressor = ICA(whiten=True)
        t0 = time()
        compressor.fit(tx, y=ty)
        newtx = compressor.transform(tx)
        runtime=time() - t0
        print newtx.shape, runtime
        distances = np.linalg.norm(tx-compressor.inverse_transform(newtx))
        print distances
    print "ica done"
    print "RP"
    for j in range(tx[1].size):
        i = j + 1
        print "===" + str(i)
        compressor = RandomProjection(n_components=i)
        t0 = time()
        compressor.fit(tx, y=ty)    
        newtx = compressor.transform(tx)
        runtime=time() - t0
        shape = newtx.shape
        print runtime, shape
    print "RP done"
    print "K-best"
    for j in range(tx[1].size):
        i = j + 1
        print "===" + str(i)
        compressor = best(add, k=i)
        t0 = time()
        compressor.fit(tx, y=ty.ravel())
        newtx = compressor.transform(tx)
        runtime=time() - t0
        shape = newtx.shape
        print runtime, shape
    print "K-best done"

def clustering(tx, ty, rx, ry, num=2):
    print "KM"
    for j in range(tx[1].size - 1):
        i = j + 1
        clf = KM(n_components=i)
        clf.fit(tx)
    print "KM done"
    print "EM"
    for j in range(tx[1].size - 1):
        i = j + 1
        clf = EM(n_components=i)
        clf.fit(tx)
    print "EM done"


if __name__ == "__main__":
    print "----BC-----"
    tx, ty, rx, ry = util.load_breast_cancer()
    dimensional(tx, ty, rx, ry, add=chi2)
    print "----V-----"
    tx, ty, rx, ry = util.load_vowel()
    dimensional(tx, ty, rx, ry, add=f_classif)

