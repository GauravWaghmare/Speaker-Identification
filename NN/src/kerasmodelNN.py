import numpy as np
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
from sklearn.decomposition import PCA
from keras.regularizers import l2, activity_l2


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


speaker_no = testFile.split("/")
speaker_no = speaker_no[-1]
speaker_no = list(speaker_no)
speaker_no = speaker_no[:-4]
speaker_no = "".join(speaker_no)
speaker_no = int(speaker_no)
print speaker_no
print 
print 


test_data = VQ.getFeaturesFromWave(testFile)  ### frames by features(34)

num_speakers = 15
train_data = load_data(direc, num_speakers)

X_train = train_data[0]
y_train = train_data[1]


print X_train.shape
# mean = np.mean(X_train, axis = 0)
# X_train -= mean
# std = np.std(X_train, axis = 0)
# X_train /= std
# cov = np.dot(X_train.T, X_train) / X_train.shape[0]
# print cov.shape
# U,S,V = np.linalg.svd(cov)
# Xrot = np.dot(X, U)
# Xrot_reduced = np.dot(X_train, U[:,:14])

# print Xrot_reduced.shape
# X_train = Xrot_reduced
# print np.amax(X_train)

# X_train = preprocessing.normalize(X_train)
# print X_train.shape

pca = PCA(n_components='mle', whiten=True) 
pca.fit(X_train)
X_train = pca.transform(X_train)


# lda = mlpy.LDA()
# print type(lda)
# lda.learn(X_train, y_train)
# print type(lda)
# print
# X_train = lda.transform(X_train)
# print "X_train"

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
# X_test -= mean
# X_test /= std
# Xtest_reduced = np.dot(X_test, U[:,:14])
# X_test = Xtest_reduced

print X_test.shape
# X_test = preprocessing.normalize(X_test)
X_test = pca.transform(X_test)

# X_test = lda.transform(X_test) #using the model to project Z
print X_test.shape
# z_labels = lda.predict(Z) #gives you the predicted label for each sample
# z_prob = lda.predict_proba(Z)
print X_train.shape[1]

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

model.fit(X_train, y_train, nb_epoch=80, validation_split= 0.2)
# score = model.evaluate(X_test, y_test, batch_size=16)
# print
modelNN = model.predict(X_test)

print modelNN.shape
print type(modelNN)


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
# print score

