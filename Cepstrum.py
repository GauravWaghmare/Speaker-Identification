import numpy as np
from scipy import signal
from scipy import fftpack
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

[fs, data] = wavfile.read("/home/gaurav/Documents/female_zero/s1.wav")

n_samples = int(fs*0.03)

data = data[:n_samples]
time_axis = np.array(range(n_samples))/fs

data_cep = fftpack.rfft(data)
data_cep = np.absolute(data_cep)
data_cep = np.log10(data_cep)
data_cep = fftpack.irfft(data_cep)

#Plot signal vs time
first = plt.subplot(211)
first.plot(time_axis, data)
first.set_title("Signal")
second = plt.subplot(212)
second.plot(time_axis, data_cep)
second.set_title("Cepstrum")
plt.show() 


