import numpy as np
# from sklearn.lda import LDA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import math
import VQ
import mlpy
from sklearn.lda import LDA
from sklearn import preprocessing
from sklearn.decomposition import PCA
# from keras.regularizers import l2, activity_l2
# from sklearn.preprocessing import OneHotEncoder
from VQ import Features
# import glob




num_speakers = 10
direc = "/home/manvi/Desktop/voicebiometric/Phoneme/new_trainset/"

featuresObj = Features(0.032, 0.016, direc, num_speakers)



train_data = featuresObj.load_data()

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




testdirec = "/home/manvi/Desktop/voicebiometric/Phoneme/Test/"
files = glob.glob(testdirec + "*.wav")


tot_positives = 0

speaker_no = 0
while speaker_no<num_speakers:
	speaker_no+=1
	print "speaker_no = " + str(speaker_no)
	print 
	print
	testFile = direc + str(speaker_no) + ".wav"
	test_data = featuresObj.getFeaturesFromWave(testFile)  ### frames by features(34)

	X_test = test_data
	X_test = pca.transform(X_test)
	modelNN = model.predict(X_test)

	positives = 0
	negatives = 0

	for u in modelNN:
		index = np.argmax(u)
		if index==speaker_no-1:
			positives +=1
		else:
			negatives +=1


	recall = float(positives*1.0/(modelNN.shape[0]))
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

