import data_util as util
import numpy as np

print "---bc--"
bc = util.load_breast_cancer()
print "train", len(bc[2]), np.bincount(bc[2])
print "test", len(bc[3]), np.bincount(bc[3])

print "---v--"
vowels = util.load_vowel()
print "train", len(vowels[2]), np.bincount(vowels[2])
print "test", len(vowels[3]), np.bincount(vowels[3])
