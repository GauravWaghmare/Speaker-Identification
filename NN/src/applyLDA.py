import sklearn
from sklearn.lda import LDA
# import sklearn.discriminant_analysis.LinearDiscriminantAnalysis
# from sklearn import discriminant_analysis

def applyLDA(X):
	# y = map(find_one, Y)
	clf = LDA()
	output = clf.fit_transform(X, y=None)
	# print output.shape
	# clf = LinearDiscriminantAnalysis()
	return output
