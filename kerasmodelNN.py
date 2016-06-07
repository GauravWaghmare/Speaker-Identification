# import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
# import feature_loader
import math
from sklearn.preprocessing import OneHotEncoder

# direc = "/home/manvi/Desktop/voicebiometric/Phoneme/trainset/"
# num_speakers = 5

# train_data, val_data, test_data = feature_loader.load_data(direc, num_speakers)

# X_train = train_data[0]
# print X_train.shape

# y_train = train_data[1]
# print y_train
# y_train = np.reshape(y_train, (y_train.shape[0], 1))
# enc = OneHotEncoder()
# y_train =enc.fit_transform(y_train).toarray()
# print y_train

# X_val = val_data[0]
# y_val = val_data[1]
# y_val = np.reshape(y_val, (y_val.shape[0], 1))
# enc = OneHotEncoder()
# y_val =enc.fit_transform(y_val).toarray()

# X_test = test_data[0]
# y_test = test_data[1]
# y_test = np.reshape(y_test, (y_test.shape[0], 1))
# enc = OneHotEncoder()
# y_test =enc.fit_transform(y_test).toarray()

# model = Sequential()
# # Dense(64) is a fully-connected layer with 64 hidden units.
# # in the first layer, you must specify the expected input data shape:
# # here, 20-dimensional vectors.
# model.add(Dense(64, input_dim=34, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(5, init='uniform'))
# model.add(Activation('softmax'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])

# model.fit(X_train, y_train, nb_epoch=20, validation_data = (X_val, y_val))
# score = model.evaluate(X_test, y_test, batch_size=16)
# print
# print model.predict(X_test)
# print
# print score




import numpy as np
import numpy
import featureExtraction
import glob, os
import scipy.io.wavfile
import scipy.cluster.vq as vq
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from scipy.spatial.distance import cdist

frame_size = 25.0/1000.0 # to convert seconds to miliseconds
frame_shift = 10.0/1000.0  # to convert seconds to miliseconds

direc = "/home/gaurav/Documents/Phoneme/trainset/"

num_speakers = 5
srno = 0
# speaker = numpy.zeros(shape=(num_speakers*4,34))

x_train = []
y_train = []

fno = 0

# Find argmin for a 2D matrix
def findMin(matrix):
	mini = matrix[0][0]
	min_ind = [0,0]
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if matrix[i][j]<mini:
				mini = matrix[i][j]
				min_ind = [i, j]
	return min_ind

while srno < num_speakers:
	srno = srno+1
	print srno
	print
	mapping = []
	# fno =0 
	directory = direc + str(srno) + "/"
	# print directory
	utterances = glob.glob(directory + "*.wav") #returns a list of names matching the argument in the directory
	cb = False
	# print utterances
	for fname in utterances:
		# print fname
		fno += 1
		print fno
		print
		fs, signal = scipy.io.wavfile.read(fname)
		window_len = frame_size*fs # Number of samples in frame_size
		sample_shift = frame_shift*fs # Number of samples shifted
		mtWin = sample_shift*10.0
		mtStep = 5.0*sample_shift
		# mtWin, mtStep, stWin, stStep)
		mtfeatures, stfeatures = featureExtraction.mtFeatureExtraction(signal, fs,mtWin , mtStep, window_len, sample_shift)
		# print len(mtfeatures)
		# print mtfeatures.shape
		features = mtfeatures
		# features = featureExtraction.stFeatureExtraction(signal, fs, window_len, sample_shift)
		# Transposing features to whiten them
		features = np.reshape(features, (features.shape[1], features.shape[0]))
		features = vq.whiten(features)
		feature_codebook = vq.kmeans2(features,8,minit='points')[0]
		if cb==False:
			codebook1 = feature_codebook
			cb = True
			x_train.append(feature_codebook)
		    y_train.append(srno)
			continue
		z = cdist(codebook1, feature_codebook)
		for i in range(8):
			x, y = findMin(z)
			mapping.append((x,y))
			z[x, :] = 65000
			z[:, y] = 65000
		mapping.sort()
		ind = map(lambda x: x[1], mapping)
		feature_codebook = feature_codebook[ind][:]
		# print z
		# print vq.vq(features, feature_codebook)
		# feature_codebook = feature_codebook.flatten()
		x_train.append(feature_codebook)
		y_train.append(srno)


# 	distanceCopy = distances[:]

# mapping = []
# map_ij = []
# idx = 1

# for i in range(8):
# 	map_ij.append([i,i])

# mapping.append(map_ij)


# print len(distances)

# while idx<len(distances):
# 	z = distances[idx]
# 	idx2 = 0
# 	map_ij = []
# 	while idx2 < 8:
# 		i,j = findMin(z)
# 		map_ij.append([i,j])
# 		m = 0
# 		while m < 8:
# 			z[i][m] = 65000
# 			m+=1
# 		m = 0
# 		while m<8:
# 			z[m][j] = 65000
# 			m+=1
# 		idx2 += 1
# 	mapping.append(map_ij)
# 	# print map_ij
# 	idx += 1

# print len(mapping)

# for i in mapping:
# 	print i

# indices  = numpy.random.permutation(10)
# print type(indices)


# i = 0
# indices = []
# shuffledDists = []
# shuffledIndi = []

# while i<len(distanceCopy):
# 	z = distanceCopy[i]
# 	indices = []
# 	map_ij = mapping[i]
# 	for j in map_ij:
# 		# print j[1]
# 		indices.append(j[1])
# 	print indices
# 	i += 1
# 	z = z[indices, :] # Good one
# 	print z
# 	shuffledIndi.append(indices)
# 	shuffledDists.append(z)


# print len(x_train)
# print len(y_train)
# print len(shuffledDists)

# x_traincopy = x_train[:]
# x_train = []
# i = 0

# while i<len(x_traincopy):
# 	x = x_traincopy[i]
# 	indices = shuffledIndi[i]
# 	print indices
# 	x = x[indices,:]
# 	x_train.append(x)
# 	i += 1



# 	X = X[indices, :]



# x_train = np.array(x_train)
# # print x_train.shape
# # y_train = np.array([0]*4+[1]*4+[2]*4+[3]*4+[4]*4)
# y_train = np.array(y_train)
# print y_train.shape
# print
# print
# y_train = np.reshape(y_train, (y_train.shape[0],1 ))
# enc = OneHotEncoder()
# y_train = enc.fit_transform(y_train).toarray()
# # print y_train
# # print y_train.shape

# x_test = []

# directory = "/home/manvi/Desktop/voicebiometric/Phoneme/testset/"
# utterances = glob.glob(directory + "*.wav")
# # print utterances
# utterances.sort()
# # print utterances
# fno = 0
# y_test = []
# while fno < len(utterances):
# 	fno += 1
# 	fname = directory + str(fno) + ".wav"
# 	fs, signal = scipy.io.wavfile.read(fname)
# 	window_len = frame_size*fs # Number of samples in frame_size
# 	sample_shift = frame_shift*fs # Number of samples shifted
# 	mtWin = sample_shift*10.0
# 	mtStep = 5.0*sample_shift
# 	# mtWin, mtStep, stWin, stStep)
# 	mtfeatures, stfeatures = featureExtraction.mtFeatureExtraction(signal, fs,mtWin , mtStep, window_len, sample_shift)
# 	print len(mtfeatures)
# 	print mtfeatures.shape
# 	features = mtfeatures

# 	# Transposing features to whiten them
# 	features = np.reshape(features, (features.shape[1], features.shape[0]))
# 	features = vq.whiten(features)
# 	feature_codebook = vq.kmeans2(features,8, minit='points')[0]
# 	# feature_codebook = feature_codebook.flatten()
# 	x_test.append(feature_codebook)
# 	y_test.append(fno)

# x_test = np.array(x_test)
# # print x_test.shape

# data_dim = 68 # Gaurav : Number of features 
# timesteps = 8 # Gaurav : Number of states 
# nb_classes = 5 # Gaurav : Number of users

# # expected input data shape: (batch_size, timesteps, data_dim)
# model = Sequential()
# model.add(LSTM(32, return_sequences=True, # Gaurav : 32 neurons in the first hidden layer
#                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(32, return_sequences=True))  # Gaurav : 32 neurons in the second hidden layer
# model.add(LSTM(32))  # Gaurav : 32 neurons in the third hidden layer
# model.add(Dense(5, activation='softmax')) # Gaurav : 10 neurons in output layer 

# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# # generate dummy training data
# # x_train = np.random.random((100, timesteps, data_dim)) # Gaurav : Sample x state vector x feature
# # y_train = np.random.random((100, nb_classes)) # Gaurav : Sample x user

# # generate dummy validation data
# # x_val = np.random.random((10, timesteps, data_dim)) # Gaurav : Sample x state vector x feature
# # y_val = np.random.random((10, nb_classes)) # Gaurav : Sample x user

# model.fit(x_train, y_train, nb_epoch=200, validation_split=0.1)

# print model.predict(x_test)

# # print score
