from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def find_one(iterable):
	for element in iterable:
		if element==1:
			return i 

def applyLDA(X, Y):
	y = map(find_one, Y)
	clf = LinearDiscriminantAnalysis()
	return clf.fit_transform(X, y)
