import numpy as np
import featureExtraction
import glob
import scipy.io.wavfile
import scipy.cluster.vq as vq
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils.visualize_util import plot

frame_size = 25.0/1000.0 # to convert seconds to miliseconds
frame_shift = 10.0/1000.0  # to convert seconds to miliseconds


direc = "/home/Gaurav/Documents/Phoneme/trainset/"


num_speakers = 5

srno = 0

speaker = numpy.zeros(shape=(num_speakers*4,34))

x_train = []

while srno < num_speakers:
	srno = srno+1
	directory = direc + str(srno) + "/"
	utterances = glob.glob(directory + "*.wav") #returns a list of names matching the argument in the directory
	for fname in utterances:
		fn = fname.split('/')
		fn = fn[-1] #take just the name of the wav file
		fs, signal = scipy.io.wavfile.read(fname)
		window_len = frame_size*fs # Number of samples in frame_size
		sample_shift = frame_shift*fs # Number of samples shifted
		features = featureExtraction.stFeatureExtraction(signal, fs, window_len, sample_shift)
		# Transposing features to whiten them
		features = np.reshape(features, (features.shape[1], features.shape[0]))
		features = vq.whiten(features)
		feature_codebook = vq.kmeans2(features,8)[0]
		x_train.append(feature_codebook)

x_train = np.array(feature_vector)


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
          validation_data=(x_val, y_val))

plot(model, to_file='model.png')