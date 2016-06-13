import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile as wavfile
import sklearn
from features import mfcc
from sklearn import svm
import features.sigproc as sigproc

[fs, data] = wavfile.read("/home/manvi/Desktop/voicebiometric/10recordings/1.wav")
print(data)

i = 0
while data[i]==0:
	i += 1

data = data[i:]

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

a = energy.tolist()
tenp = int(0.1*rows)
a.sort() 
a = np.array(a)
low_index = a[:tenp, 1]
high_index = a[rows-tenp:, 1]
index = np.append(low_index, high_index)

time_axis = np.array(range(len(data)))/fs

mfcc_feat = mfcc(data, samplerate=fs, appendEnergy=False)
feature_vector = []
for i in range(tenp*2):
	feature_vector.append(mfcc_feat[index[i]])
feature_vector = np.array(feature_vector)
target_vector = np.vstack((np.ones((tenp, 1)), np.zeros((tenp,1))))

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