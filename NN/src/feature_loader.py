import VQ
import numpy

def load_data(direc, num_speakers):
	X, Y = VQ.getFeatureMatrix(direc, num_speakers)
	indices  = numpy.random.permutation(Y.shape[0])

	X = X[indices, :]
	Y = Y[indices]

	total_rows = Y.shape[0]
	train_data_rows = int(total_rows)
	# val_data_rows = int(0.1*total_rows)
	# test_data_rows = int(0.1*total_rows)

	train_x = X[0:train_data_rows+1, :]
	train_y = Y[0:train_data_rows+1]
	train_data = (train_x, train_y)

	# val_x = X[train_data_rows+1: train_data_rows+val_data_rows+1, :]
	# val_y = Y[train_data_rows+1: train_data_rows+val_data_rows+1]
	# val_data = (val_x, val_y)

	# # test_x = X[train_data_rows+val_data_rows+1: train_data_rows+val_data_rows+test_data_rows+1, :]
	# # test_y = Y[train_data_rows+val_data_rows+1: train_data_rows+val_data_rows+test_data_rows+1]
	# # test_data = (test_x, test_y)

	return train_data



# direc = "/home/manvi/Desktop/voicebiometric/Phoneme/trainset/"
# num_speakers = 5
# a = load_data(direc, num_speakers)
# print len(a)
# print