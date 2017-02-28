# from __future__ import print_function
import featureExtraction
import glob
import numpy
import scipy.io.wavfile
from keras.regularizers import l2, activity_l2
import math
import LPC
import Removesilence as rs
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
# from features import mfcc

# print("Successfully imported features")

class Features(object):
	"""docstring for Features"""
	
	def __init__(self, frame_size, frame_shift, direc, num_speakers):
		self.frame_size = float(frame_size)
		self.frame_shift = float(frame_shift)
		self.direc = direc
		self.num_speakers = int(num_speakers)
	
	def getTrainingMatrix(self):
		srno = 0
		flag = False
		fno =0
		while (srno < self.num_speakers):
			srno = srno+1
			print "\nsrno = " + str(srno)
			directory = self.direc + str(srno) + "/"
			utterances = glob.glob(directory + "*.wav")

			for fname in utterances:
				fn = fname.split('/')
				fn = fn[-1]
				fno = fno + 1
				print
				print fn
				print fname
				featuresT = self.getFeaturesFromWave(fname)

				if flag==False:
					c = featuresT
					flag = True
					y = numpy.ones(shape=(featuresT.shape[0],))
				else:
					c = numpy.concatenate((c, featuresT), axis = 0)
					y1 = numpy.ones(shape=(featuresT.shape[0],))
					y1.fill(srno)
					y = numpy.concatenate((y,y1), axis = 0)

		return (c, y)


	def getFeaturesFromWave(self, fname):
		fs, signal = scipy.io.wavfile.read(fname)
		window_len = self.frame_size*fs 			# Number of samples in frame_size
		sample_shift = self.frame_shift*fs 		# Number of samples shifted
		try:
			if signal.shape[1]:
				signal = numpy.mean(signal, axis=1)
		except:
			print "single column"

		segmentLimits = rs.silenceRemoval(signal, fs, self.frame_size, self.frame_shift)
		segmentLimits = numpy.asarray(segmentLimits)
		data = rs.nonsilentRegions(segmentLimits, fs, signal)

		stfeatures = featureExtraction.stFeatureExtraction(data, fs, window_len, sample_shift )
		lpc = LPC.extract((fs, data))
		featuresT = stfeatures.transpose()
		featuresT = numpy.concatenate((featuresT, lpc), axis = 1)
		return featuresT


	def load_data(self):
		X, Y = self.getTrainingMatrix()
		indices  = numpy.random.permutation(Y.shape[0])
		X = X[indices, :]
		Y = Y[indices]
		train_data_rows = int(Y.shape[0])

		train_x = X[0:train_data_rows+1, :]
		train_y = Y[0:train_data_rows+1]
		train_data = (train_x, train_y)
		return train_data




class Train(object):
	"""docstring for Train"""
	frame_size = 0.032
	frame_shift = 0.016

	def __init__(self, num_speakers, directory, frame_size, frame_shift):
		print "Training"
		self.num_speakers = num_speakers
		self.directory = direc
		Train.frame_size = frame_size
		Train.frame_shift = frame_shift
		self.featuresObj = Features(frame_size,frame_shift,directory,num_speakers)
		self.model = Sequential()


	def Pca(self, X_train):	
		pca = PCA(n_components='mle', whiten = True) 
		pca.fit(X_train)
		X_train = pca.transform(X_train)
		return (X_train, pca)


	def encodeY(self, y_train):
		y_train = numpy.reshape(y_train, (y_train.shape[0], 1))
		enc = OneHotEncoder()
		y_train =enc.fit_transform(y_train).toarray()
		return y_train


	def train(self, epochs = 60, activation_fn = 'glorot_normal'):
		print "Train function called."
		train_data = self.featuresObj.load_data()
		X_train = train_data[0]
		y_train = train_data[1]
		print X_train.shape

		X_train, pca = self.Pca(X_train)
		y_train = self.encodeY(y_train)

		print X_train.shape
		print y_train.shape
		print 
		self.model.add(Dense(64, input_dim=X_train.shape[1] , init=activation_fn))
		self.model.add(Activation('tanh'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(self.num_speakers, init=activation_fn))
		self.model.add(Activation('softmax'))

		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='categorical_crossentropy',
		              optimizer=sgd,
		              metrics=['accuracy'])

		# print "/n"
		# print "Training starts"
		# print "/n"

		self.model.fit(X_train, y_train, nb_epoch=epochs, validation_split= 0.2)

		# print "/n"
		# print "Training ends"
		# print "/n"

		print "Train function exits"

		return pca


	def test(self, testdirec):
		print "Test function called."
		pca = self.train()
		files = glob.glob(testdirec + "*.wav")
		tot_positives = 0
		speaker_no = 0

		while (speaker_no < self.num_speakers):
			speaker_no += 1
			testFile = testdirec + str(speaker_no) + ".wav"
			print testFile
			print

			test_data = self.featuresObj.getFeaturesFromWave(testFile)  ### frames by features(34)

			X_test = test_data
			X_test = pca.transform(X_test)
			modelNN = self.model.predict(X_test)

			positives = 0
			negatives = 0

			for u in modelNN:
				index = numpy.argmax(u)
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
			sumRows = numpy.array(sumRows)
			sumRows /= modelNN.shape[0]
			print sumRows
			print

			index = numpy.argmax(sumRows)
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
		print "Test function exits"
		return tot_positives




direc = "/home/gaurav/Downloads/new_trainset/"
testdirec = "/home/gaurav/Downloads/new_trainset/Test/"
# print
# print "Training starts"
t = Train(num_speakers=10, directory = direc, frame_size=0.032, frame_shift=0.016)
# print "Training ends"
# print
# pca = t.train()
tot_positives = t.test(testdirec)
print tot_positives