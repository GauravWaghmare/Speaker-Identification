import featureExtraction
import glob
import numpy
import scipy.io.wavfile
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l2, activity_l2
import LPC
import utils
import Removesilence as rs
import os


class Features(object):
	"""docstring for Features"""
	num_speakers = 0
	mapping = {}
	def __init__(self, frame_size, frame_shift):
		self.frame_size = float(frame_size)
		self.frame_shift = float(frame_shift)
		# self.direc = direc
		# num_speakers = int(num_speakers)
	
	def getTrainingMatrix(self, direc):
		srno = 0
		flag = False
		fno =0
		for user_directory in os.listdir(direc):
			print
			print "username = " + str(user_directory)
			phone_number = user_directory.split("-")
			phone_number = phone_number[0]
			phone_number = int(phone_number)
			print phone_number
			print
			user_directory_path = os.path.join(direc, user_directory)

			if os.path.isdir(user_directory_path):
				Features.num_speakers += 1
				Features.mapping[Features.num_speakers] = phone_number
				srno += 1
				for file in os.listdir(user_directory_path):
					print "\nfile_name = " + str(file)
					fname = os.path.join(user_directory_path, file)
					if os.path.isfile(fname):
						fn = fname.split('/')
						fn = fn[-1]
						if fn[-4:]=='.wav':
							featuresT = self.getFeaturesFromWave(fname)
							if flag==False:
								c = featuresT
								flag = True
								y = numpy.ones(shape=(featuresT.shape[0],))
								y.fill(Features.num_speakers)
							else:
								c = numpy.concatenate((c, featuresT), axis = 0)
								y1 = numpy.ones(shape=(featuresT.shape[0],))
								y1.fill(Features.num_speakers)
								y = numpy.concatenate((y,y1), axis = 0)
						else:
							print "file is not an audio file"

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
		lpc = LPC.extract((fs, data), win_len = 32, win_shift = 16)
		featuresT = stfeatures.transpose()
		featuresT = numpy.concatenate((featuresT, lpc), axis = 1)
		return featuresT


	def load_data(self, directory):
		X, Y = self.getTrainingMatrix(directory)
		indices  = numpy.random.permutation(Y.shape[0])
		X = X[indices, :]
		Y = Y[indices]
		train_data_rows = int(Y.shape[0])

		train_x = X[0:train_data_rows+1, :]
		train_y = Y[0:train_data_rows+1]
		train_data = (train_x, train_y)
		return train_data