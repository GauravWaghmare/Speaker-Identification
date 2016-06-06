import numpy as np
import numpy
import featureExtraction
import glob, os
import scipy.io.wavfile
import scipy.cluster.vq as vq
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense

frame_size = 25.0/1000.0 # to convert seconds to miliseconds
frame_shift = 10.0/1000.0  # to convert seconds to miliseconds

direc = "/home/gaurav/Documents/Phoneme/trainset/"

num_speakers = 5
srno = 0
speaker = numpy.zeros(shape=(num_speakers*4,34))

x_train = []

while srno < num_speakers:
	srno = srno+1
	directory = direc + str(srno) + "/"
	# print directory
	utterances = glob.glob(directory + "*.wav") #returns a list of names matching the argument in the directory
	# print utterances
	for fname in utterances:
		# print fname
		fs, signal = scipy.io.wavfile.read(fname)
		window_len = frame_size*fs # Number of samples in frame_size
		sample_shift = frame_shift*fs # Number of samples shifted
		features = featureExtraction.stFeatureExtraction(signal, fs, window_len, sample_shift)
		# Transposing features to whiten them
		features = np.reshape(features, (features.shape[1], features.shape[0]))
		features = vq.whiten(features)
		feature_codebook = vq.kmeans2(features,8)[0]
		x_train.append(feature_codebook)

x_train = np.array(x_train)
# print x_train.shape
y_train = np.array([0]*4+[1]*4+[2]*4+[3]*4+[4]*4)
y_train = np.reshape(y_train, (20,1))
enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()
# print y_train
# print y_train.shape

x_test = []

directory = direc + "test/"
utterances = glob.glob(directory + ".wav")
for fname in utterances:
	fs, signal = scipy.io.wavfile.read(fname)
	window_len = frame_size*fs # Number of samples in frame_size
	sample_shift = frame_shift*fs # Number of samples shifted
	features = featureExtraction.stFeatureExtraction(signal, fs, window_len, sample_shift)
	# Transposing features to whiten them
	features = np.reshape(features, (features.shape[1], features.shape[0]))
	features = vq.whiten(features)
	feature_codebook = vq.kmeans2(features,8)[0]
	x_test.append(feature_codebook)

x_test = np.array(x_test)

data_dim = 34 # Gaurav : Number of features 
timesteps = 8 # Gaurav : Number of states 
nb_classes = 5 # Gaurav : Number of users

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True, # Gaurav : 32 neurons in the first hidden layer
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # Gaurav : 32 neurons in the second hidden layer
model.add(LSTM(32))  # Gaurav : 32 neurons in the third hidden layer
model.add(Dense(5, activation='softmax')) # Gaurav : 10 neurons in output layer 

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# generate dummy training data
# x_train = np.random.random((100, timesteps, data_dim)) # Gaurav : Sample x state vector x feature
# y_train = np.random.random((100, nb_classes)) # Gaurav : Sample x user

# generate dummy validation data
# x_val = np.random.random((10, timesteps, data_dim)) # Gaurav : Sample x state vector x feature
# y_val = np.random.random((10, nb_classes)) # Gaurav : Sample x user

model.fit(x_train, y_train, nb_epoch=5,
          validation_split=0.2)

model.predict(x_test)