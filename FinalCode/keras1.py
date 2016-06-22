import numpy
# from sklearn.lda import LDA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import math
import VQ
import mlpy
# from sklearn.lda import LDA
from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras.regularizers import l2, activity_l2
from sklearn.preprocessing import OneHotEncoder
from VQ import Features
import glob



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

		self.model.fit(X_train, y_train, nb_epoch=epochs, validation_split= 0.2)

		return pca


	def test(self, testdirec):
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
			# print modelNN

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
			print sumRows[index]
			
			if index==speaker_no-1:
				if sumRows[index]<0.4:
					print ' true but less than 0.4' 
					print
					print
				tot_positives+=1
				print "true"
			else:
				print "false"


			print 
			print

		print "total number of correct answers = " + str(tot_positives)
		print 
		print
		return tot_positives




direc = "/home/manvi/Desktop/voicebiometric/Phoneme/new_trainset/"
testdirec = "/home/manvi/Desktop/voicebiometric/Phoneme/Test/"

t = Train(num_speakers=14, directory = direc, frame_size=0.032, frame_shift=0.016)

pca = t.train()
tot_positives = t.test(testdirec)
print tot_positives
# num_speakers = 10

# featuresObj = Features(0.032, 0.016, direc, num_speakers)
# train_data = featuresObj.load_data()
# X_train = train_data[0]
# y_train = train_data[1]
# print X_train.shape

# # mean = numpy.mean(X_train, axis = 0)
# # X_train -= mean
# # std = numpy.std(X_train, axis = 0)
# # X_train /= std
# # cov = numpy.dot(X_train.T, X_train) / X_train.shape[0]

# pca = PCA(n_components='mle', whiten = True) 
# pca.fit(X_train)
# X_train = pca.transform(X_train)


# print X_train.shape
# print
# print

# y_train = numpy.reshape(y_train, (y_train.shape[0], 1))
# enc = OneHotEncoder()
# y_train =enc.fit_transform(y_train).toarray()


# model = Sequential()
# model.add(Dense(64, input_dim=X_train.shape[1] , init='glorot_normal'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# # model.add(Dense(64, init='uniform'))
# # model.add(Activation('tanh'))
# # model.add(Dropout(0.5))
# # model.add(Dense(10, init='he_normal'))
# # model.add(Activation('tanh'))
# # model.add(Dropout(0.5))

# model.add(Dense(num_speakers, init='glorot_normal'))
# model.add(Activation('softmax'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])

# model.fit(X_train, y_train, nb_epoch=60, validation_split= 0.2)




# testdirec = "/home/manvi/Desktop/voicebiometric/Phoneme/Test/"
# files = glob.glob(testdirec + "*.wav")


# tot_positives = 0

# speaker_no = 0
# while speaker_no<num_speakers:
# 	speaker_no+=1
# 	print "speaker_no = " + str(speaker_no)
# 	print 
# 	print
# 	testFile = direc + str(speaker_no) + ".wav"
# 	test_data = featuresObj.getFeaturesFromWave(testFile)  ### frames by features(34)

# 	X_test = test_data
# 	X_test = pca.transform(X_test)
# 	modelNN = model.predict(X_test)

# 	positives = 0
# 	negatives = 0

# 	for u in modelNN:
# 		index = numpy.argmax(u)
# 		if index==speaker_no-1:
# 			positives +=1
# 		else:
# 			negatives +=1


# 	recall = float(positives*1.0/(modelNN.shape[0]))
# 	print "recall = "  + str(recall)

# 	sumRows = []
# 	for i in xrange(0,modelNN.shape[1]):
# 		sumRows.append(0)

# 	for i in modelNN:
# 		j = 0
# 		while j < len(i):
# 			sumRows[j] = sumRows[j] + i[j]
# 			j+=1

# 	print sumRows
# 	sumRows = numpy.array(sumRows)
# 	sumRows /= modelNN.shape[0]
# 	print sumRows
# 	print

# 	index = numpy.argmax(sumRows)
# 	if index==speaker_no-1:
# 		tot_positives+=1
# 		print "true"
# 	else:
# 		print "false"
# 	print 
# 	print

# print "total number of correct answers = " + str(tot_positives)
# print 
# print
# 	# print score

