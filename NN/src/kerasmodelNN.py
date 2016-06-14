import numpy as np
# import np
from sklearn.lda import LDA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
# import feature_loader
import math
from sklearn.preprocessing import OneHotEncoder
import VQ
import applyLDA as al
import mlpy
from sklearn.lda import LDA
from sklearn import preprocessing

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



direc = "/home/manvi/Desktop/voicebiometric/Phoneme/trainset/"

testFile = "/home/manvi/Desktop/voicebiometric/Phoneme/testset/2.wav"

test_data = VQ.getFeaturesFromWave(testFile)  ### frames by features(34)

num_speakers = 15

train_data = load_data(direc, num_speakers)

X_train = train_data[0]
y_train = train_data[1]
X_train = preprocessing.scale(X_train)
print X_train.shape
# lda = LDA() #creating a LDA object
# lda = lda.fit(X_train, y_train) #learning the projection matrix
# X_train = lda.transform(X_train) #using the model to project X 
print X_train.shape
print
print
y_train = np.reshape(y_train, (y_train.shape[0], 1))
enc = OneHotEncoder()
y_train =enc.fit_transform(y_train).toarray()

X_test = test_data
X_test = preprocessing.scale(X_test)
# X_test = lda.transform(X_test) #using the model to project Z
print X_test.shape
# z_labels = lda.predict(Z) #gives you the predicted label for each sample
# z_prob = lda.predict_proba(Z)

model = Sequential()
model.add(Dense(70, input_dim=X_train.shape[1] , init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
# model.add(Dense(100, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
model.add(Dense(70, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_speakers, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=50, validation_split= 0.3)
# score = model.evaluate(X_test, y_test, batch_size=16)
# print
modelNN = model.predict(X_test)


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
# print score