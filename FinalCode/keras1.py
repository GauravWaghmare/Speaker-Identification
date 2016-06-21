import numpy as np
from sklearn.lda import LDA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import math
import VQ
import applyLDA as al
import mlpy
from sklearn.lda import LDA
from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras.regularizers import l2, activity_l2
from sklearn.preprocessing import OneHotEncoder


def load_data(direc, num_speakers):
	X, Y = VQ.getFeatureMatrix(direc, num_speakers)
	indices  = np.random.permutation(Y.shape[0])

	X = X[indices, :]
	Y = Y[indices]

	total_rows = Y.shape[0]
	train_data_rows = int(total_rows)

	train_x = X[0:train_data_rows+1, :]
	train_y = Y[0:train_data_rows+1]
	train_data = (train_x, train_y)

	return train_data



direc = "/home/manvi/Desktop/voicebiometric/Phoneme/new_trainset/"

num_speakers = 10
train_data = load_data(direc, num_speakers)

X_train = train_data[0]
y_train = train_data[1]


print X_train.shape
# mean = np.mean(X_train, axis = 0)
# X_train -= mean
# std = np.std(X_train, axis = 0)
# X_train /= std
# cov = np.dot(X_train.T, X_train) / X_train.shape[0]


pca = PCA(n_components='mle', whiten = True) 
pca.fit(X_train)
X_train = pca.transform(X_train)


print X_train.shape
print
print
y_train = np.reshape(y_train, (y_train.shape[0], 1))
enc = OneHotEncoder()
y_train =enc.fit_transform(y_train).toarray()



model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1] , init='glorot_normal'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(10, init='he_normal'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))

model.add(Dense(num_speakers, init='glorot_normal'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=60, validation_split= 0.2)





import glob

direc = "/home/manvi/Desktop/voicebiometric/Phoneme/Test/"
files = glob.glob(direc + "*.wav")


tot_positives = 0

speaker_no = 0
while speaker_no<num_speakers:
	speaker_no+=1
	print "speaker_no = " + str(speaker_no)
	print 
	print
	testFile = direc + str(speaker_no) + ".wav"
	test_data = VQ.getFeaturesFromWave(testFile)  ### frames by features(34)

	X_test = test_data
	# X_test -= mean
	# X_test /= std

	# print X_test.shape
	X_test = pca.transform(X_test)

	# X_test = lda.transform(X_test) #using the model to project Z
	# print X_test.shape
	# z_labels = lda.predict(Z) #gives you the predicted label for each sample
	# z_prob = lda.predict_proba(Z)

	modelNN = model.predict(X_test)

	# print modelNN.shape
	# print type(modelNN)


	positives = 0
	negatives = 0

	for u in modelNN:
		index = np.argmax(u)
		if index==speaker_no-1:
			positives +=1
		else:
			negatives +=1


	recall = float(positives*1.0/(modelNN.shape[0]))
	# prec = float(positives*1.0/(positives+negatives))
	print "recall = "  + str(recall)


	sumRows = []
	for i in xrange(0,modelNN.shape[1]):
		sumRows.append(0)

	for i in modelNN:
		j = 0
		while j < len(i):
			sumRows[j] = sumRows[j] + i[j]
			j+=1

	print sumRows
	sumRows = np.array(sumRows)
	sumRows /= modelNN.shape[0]
	print sumRows
	print

	index = np.argmax(sumRows)
	if index==speaker_no-1:
		tot_positives+=1
		print "true"
	else:
		print "false"
	print 
	print

print "total number of correct answers = " + str(tot_positives)
print 
print
	# print score

