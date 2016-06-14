from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def find_one(iterable):
	for element in iterable:
		i = 0
		while element!=1:
			i += 1
		return i 

def applyLDA(X, Y):
	y = map(find_one, Y)
	clf = LinearDiscriminantAnalysis()
	return clf.fit_transform(X, y)
