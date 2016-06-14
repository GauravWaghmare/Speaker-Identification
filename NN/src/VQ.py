from sklearn.lda import LDA
import featureExtraction
import glob
import numpy
import scipy.io.wavfile
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stats
from features import mfcc
from sklearn import cluster
import Removesilence as rs


frame_size = 35.0/1000.0 # to convert seconds to miliseconds
frame_shift = 10.0/1000.0  # to convert seconds to miliseconds



direc = "/home/manvi/Desktop/voicebiometric/Phoneme/trainset/"


def getFeatureMatrix(direc, num_speakers):
	# num_speakers = 5
	srno = 0
	flag = False

	fno =0
	row = 0
	flag = False

	while srno < num_speakers:
		srno = srno+1
		print "\nsrno = " + str(srno)
		directory = direc + str(srno) + "/"
		utterances = glob.glob(directory + "*.wav")
		# flag_2 = False

		for fname in utterances:
			fn = fname.split('/')
			fn = fn[-1]
			fno = fno + 1
			row = row+1
			fs, signal = scipy.io.wavfile.read(fname)
			window_len = frame_size*fs # Number of samples in frame_size
			sample_shift = frame_shift*fs # Number of samples shifted

			segmentLimits = rs.silenceRemoval(signal, fs, frame_size, frame_shift)
			segmentLimits = numpy.asarray(segmentLimits)
			data = rs.nonsilentRegions(segmentLimits, fs, signal)

			features, stfeatures = featureExtraction.mtFeatureExtraction(data, fs, sample_shift*10.0, 2.0*sample_shift, window_len, sample_shift )

			featuresT = features.transpose()
			print featuresT.shape
			print
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




def getFeaturesFromWave(fname, frame_size = 0.035, frame_shift = 0.010):
	fs, signal = scipy.io.wavfile.read(fname)
	window_len = frame_size*fs # Number of samples in frame_size
	sample_shift = frame_shift*fs # Number of samples shifted

	segmentLimits = rs.silenceRemoval(signal, fs, frame_size, frame_shift)
	segmentLimits = numpy.asarray(segmentLimits)
	data = rs.nonsilentRegions(segmentLimits, fs, signal)

	features, stfeatures = featureExtraction.mtFeatureExtraction(data, fs, sample_shift*10.0, 5.0*sample_shift, window_len, sample_shift )
	featuresT = features.transpose()
	return featuresT
