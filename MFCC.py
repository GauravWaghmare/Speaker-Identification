import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftpack
import scipy.stats as stats
from features import mfcc
from sklearn import cluster

#[fs, data] = wavfile.read("/home/gaurav/Downloads/Gaurav_Manvi_recs/1.wav")

#mfcc_feat = mfcc(data, samplerate=fs)
#print(mfcc_feat.shape)
#mfcc_mean = np.mean(mfcc_feat, axis=0)
#print(mfcc_mean)
#print(mfcc_mean.shape)
#mfcc_var = np.std(mfcc_feat, axis=0)
#print(mfcc_var)

mfcc_var = []
for i in range(1, 33, 1):
    #[fs, data] = wavfile.read("/home/gaurav/Downloads/Gaurav_Manvi_recs/"+str(i)+".wav")
    [fs, data] = wavfile.read("/home/gaurav/Documents/voice_samples/"+str(i)+".wav")
    mfcc_feat = mfcc(data, samplerate=fs)
    mfcc_mean = np.mean(mfcc_feat, axis=0)
    mfcc_var.append(mfcc_mean)
mfcc_var = np.array(mfcc_var)
print(mfcc_var.shape)

#print(stats.variation(mfcc_var, axis=0))

X_iris = mfcc_var
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(X_iris)
print(k_means.labels_)