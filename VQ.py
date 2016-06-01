from sklearn.lda import LDA
import featureExtraction
import glob
import numpy
import scipy.io.wavfile
import scipy.cluster.vq as vq
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stats
from features import mfcc
from sklearn import cluster


frame_size = 25.0/1000.0 # to convert seconds to miliseconds
frame_shift = 10.0/1000.0  # to convert seconds to miliseconds


direc = "/home/manvi/Desktop/voicebiometric/Phoneme/trainset/"


num_speakers = 5

srno = 0

speaker = numpy.zeros(shape=(num_speakers*4,34))


y = []
d3_y = []

fno =0
row = 0

speakers = []


while srno < num_speakers:
	srno = srno+1
	print "\nsrno = " + str(srno)
	directory = direc + str(srno) + "/"
	utterances = glob.glob(directory + "*.wav") #returns a list of names matching the argument in the directory
	# feature_vector = numpy.zeros(shape=(34,1))
	flag = False
	# flag_2 = False

	for fname in utterances:
		fn = fname.split('/')
		fn = fn[-1] #take just the name of the wav file
		# print fn 
		# print "fno = " + str(fno)
		fno = fno + 1
		row = row+1
		y.append(srno-1)
		fs, signal = scipy.io.wavfile.read(fname)
		window_len = frame_size*fs # Number of samples in frame_size
		sample_shift = frame_shift*fs # Number of samples shifted
		features = featureExtraction.stFeatureExtraction(signal, fs, window_len, sample_shift)
		# print "features.shape = " + str(features.shape)
		i = 0

		if flag == False:
			flag = True
			feature_vector = features
			print "feature vector " + feature_vector.shape
			# print
			# i = 1
		else:
			# print "arbitajvhfibvvdbli"
			feature_vector = numpy.concatenate((feature_vector, features ), axis = 1)
			print "final feature vector " + feature_vector.shape #Is feature_vector featurexframe or framexfeature? 
	d3_y.append(feature_vector)
	# print "feature_vector = " + str(feature_vector.shape)
	# print "speakers = " + str(speakers.shape)
	# print
	# speakers = numpy.append(speakers, feature_vector)


d3_y = numpy.array(d3_y)
print d3_y.shape
# print y

# Whitening every speaker's data before VQing
for i in range(d3_y.shape[0]):
	d3_y[i,:,:] = vq.whiten(d3_y[i,:,:])

# Calulating codebooks for every user
codebook = []
for i in range(d3_y.shape[0]):
	# kmeans2 function takes in the argument as MxN, M is the no. of observations and N is the number of features
	# I guess we'll have to take transpose or something of the d3_y matrix
	codebook.append(vq.kmeans2(d3_y[i,:,:],8)[0])

codebook = np.array(codebook) 


