from sklearn.svm import SVC

import data_util as util

bc_data_train, bc_data_test, bc_target_train, bc_target_test = util.load_breast_cancer()
v_data_train, v_data_test, v_target_train, v_target_test = util.load_vowel()
KERNELS = ["linear", "rbf", "rbf", "rbf", "rbf", "rbf", "rbf", "poly", "poly", "poly", "poly"]
GAMMA = [.001, .01, .5, 1., 1.25, 1.5]

def svm():
	print "---bc---"
	d = 2
	g = 0
	for kernel in KERNELS:
		if kernel == "poly":
			_svm(bc_data_train, bc_data_test, bc_target_train, bc_target_test, kernel, degree=d)
			d += 1
		elif kernel == "rbf":
			_svm(bc_data_train, bc_data_test, bc_target_train, bc_target_test, kernel, gamma=GAMMA[g])
			g += 1
		else:
			_svm(bc_data_train, bc_data_test, bc_target_train, bc_target_test, kernel)


	print "---v---"
	d = 2
	g = 0
	for kernel in KERNELS:
		if kernel == "poly":
			_svm(v_data_train, v_data_test, v_target_train, v_target_test, kernel, degree=d)
			d += 1
		elif kernel == "rbf":
			_svm(v_data_train, v_data_test, v_target_train, v_target_test, kernel, gamma=GAMMA[g])
			g += 1
		else:
			_svm(v_data_train, v_data_test, v_target_train, v_target_test, kernel)




def _svm(data, data_test, target, target_test, kernel, degree=None, gamma=None):
	if degree and kernel == "poly":
		clf = SVC(kernel=kernel, degree=degree)
	elif gamma and kernel == "rbf":
		clf = SVC(kernel=kernel, gamma=gamma)
	else:
		clf = SVC(kernel=kernel)
	clf.fit(data, target)
	train_score = clf.score(data, target)
	test_score = clf.score(data_test, target_test)
	print kernel, degree, gamma, train_score, test_score


if __name__ == "__main__":
	svm()
