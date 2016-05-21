import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile as wavfile
import sklearn
from features import mfcc
from sklearn import svm
import features.sigproc as sigproc

[fs, data] = wavfile.read("/home/gaurav/Downloads/Gaurav_Manvi_recs/1.wav")

data = np.mean(data, axis=1)
# print(data)

i = 0
while data[i]==0:
	i += 1

data = data[i:]

# def short_term_energy(speech, samplerate, win_step, win_len):
# 	win_step *= samplerate
# 	win_len *= samplerate  
# 	y = speech.copy()
# 	win = np.hamming(win_len)
# 	print(win.shape)
# 	sum1 = 0
# 	num_of_frames = (math.floor(len(y)/win_step) - math.ceil(win_len/win_step))
# 	print(num_of_frames)
# 	print(num_of_frames)
# 	energy = []
# 	jj = 1
# 	i = 0
# 	while i < num_of_frames:
# 		j = (((i-1)*win_step)+1)
# 		while j< (((i-1)*win_step) + win_len):
# 			y[j] = y[j]*win[jj]
# 			jj+=1
# 			yy = y[j]*y[j]
# 			sum1 = sum1 + yy
# 			j += 1
# 		energy.append([sum1, i])
# 		sum1 = 0
# 		jj = 1
# 		i += 1
# 	w = 0
# 	c = energy
# 	print len(energy)
# 	return [num_of_frames, energy]

winlen = fs*0.025
winstep = fs*0.01

a = sigproc.framesig(data, winlen, winstep)
# print(a.shape)
rows = a.shape[0]
# print(rows)
win = np.hamming(winlen).reshape((1, winlen))
# print(win.shape)
win = np.tile(win, (rows, 1))
# print(win.shape)
a_win = a*win
a_win2 = a_win*a_win
energy1 = np.sum(a_win2, axis=1)
ins = np.array(range(rows))
energy = np.zeros((rows,2))
energy[:,0] = energy1
energy[:,1] = ins
# print("energy shape ", energy.shape)
# print("energy ", energy)
# print(energy.shape)

# [n, a] = short_term_energy(data, fs, 0.01, 0.025)
# print(n)
a = energy.tolist()
tenp = int(0.1*rows)
# print("tenp ", tenp)
a.sort() 
a = np.array(a)
# print(a.shape)
low_index = a[:tenp, 1]
high_index = a[rows-tenp:, 1]
index = np.append(low_index, high_index)
# print("index ", index)
# print(len(low_index))
# print(len(high_index))
# print(index.shape)
time_axis = np.array(range(len(data)))/fs

# plt.figure(1)
# signal = plt.subplot(211)
# signal.plot(time_axis, data)
# signal.set_title("Signal")
# energy1 = plt.subplot(212)
# energy1.plot(a)
# energy1.set_title("Energy")
# plt.show()

mfcc_feat = mfcc(data, samplerate=fs)
# print("mfcc shape", mfcc_feat.shape)
feature_vector = []
for i in range(tenp*2):
	feature_vector.append(mfcc_feat[index[i]])
feature_vector = np.array(feature_vector)
# print("feature vector shape", feature_vector.shape)
target_vector = np.vstack((np.ones((tenp, 1)), np.zeros((tenp,1))))
# print("target vector ", target_vector.shape)

clf = svm.SVC()
clf.fit(feature_vector, target_vector)
sil = clf.predict(mfcc_feat)

plt.figure(1)
signal = plt.subplot(311)
signal.plot(data)
signal.set_title("Signal")
energyplot = plt.subplot(312)
energyplot.plot(energy)
energyplot.set_title("Energy")
silent = plt.subplot(313)
silent.plot(sil)
silent.set_title("Silence")
plt.show()