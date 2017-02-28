from sklearn.lda import LDA
import featureExtraction
import glob
import numpy
import scipy.io.wavfile
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stats
# from features import mfcc
from sklearn import cluster
import Removesilence as rs
from feature import mix_feature
from keras.regularizers import l2, activity_l2
import LPC


frame_size = 32.0/1000.0 # to convert seconds to miliseconds
frame_shift = 16.0/1000.0  # to convert seconds to miliseconds



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

		for fname in utterances:
			fn = fname.split('/')
			fn = fn[-1]
			fno = fno + 1
			row = row+1
			print
			print fn

			try:
				fs, signal = scipy.io.wavfile.read(fname)

				print fs
				window_len = frame_size*fs # Number of samples in frame_size
				sample_shift = frame_shift*fs # Number of samples shifted

				# print window_len
				# print sample_shift
				# print
				# print signal.shape
				# print signal.shape[0]

				try:
					if signal.shape[1]:
						signal = numpy.mean(signal, axis=1)
				except:
					print "single column"

				print 

				print signal.shape
				# data = signal
				segmentLimits = rs.silenceRemoval(signal, fs, frame_size, frame_shift)
				segmentLimits = numpy.asarray(segmentLimits)
				data = rs.nonsilentRegions(segmentLimits, fs, signal)

				print data.shape
				
				print "get lpc"
				lpc = LPC.extract((fs, data))
				print "LPC"
				print type(lpc)

				stfeatures = featureExtraction.stFeatureExtraction(data, fs, window_len, sample_shift )
				print stfeatures.shape

				# mtfeatures, stfeatures = featureExtraction.mtFeatureExtraction(data, fs, sample_shift*10.0, 2.0*sample_shift, window_len, sample_shift )

				featuresT = stfeatures.transpose()
				featuresT = numpy.concatenate((featuresT, lpc), axis = 1)
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
			except Exception, e:
				# print fn 
				raise e
				print "it produced an error"

	return (c, y)




def getFeaturesFromWave(fname, frame_size = 0.032, frame_shift = 0.016):
	fs, signal = scipy.io.wavfile.read(fname)
	window_len = frame_size*fs # Number of samples in frame_size
	sample_shift = frame_shift*fs # Number of samples shifted
	try:
		if signal.shape[1]:
			signal = numpy.mean(signal, axis=1)
	except:
		print "single column"


	segmentLimits = rs.silenceRemoval(signal, fs, frame_size, frame_shift)
	segmentLimits = numpy.asarray(segmentLimits)
	data = rs.nonsilentRegions(segmentLimits, fs, signal)

	# data = signal
	# featuresT = mix_feature((fs, data))
	stfeatures = featureExtraction.stFeatureExtraction(data, fs, window_len, sample_shift )
	lpc = LPC.extract((fs, data))
	featuresT = stfeatures.transpose()
	featuresT = numpy.concatenate((featuresT, lpc), axis = 1)
	# mtfeatures, stfeatures = featureExtraction.mtFeatureExtraction(data, fs, sample_shift*10.0, 5.0*sample_shift, window_len, sample_shift )
	# featuresT = stfeatures.transpose()
	return featuresT