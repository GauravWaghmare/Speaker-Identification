import featureExtraction
import glob
import numpy
import scipy.io.wavfile
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l2, activity_l2
import LPC
# import mfcc
# from features import LPC
import utils
from features import mfcc
# from features import utils
import Removesilence as rs


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

