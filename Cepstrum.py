import numpy as np
from scipy import signal
from scipy import fftpack
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

[fs, data] = wavfile.read("/home/gaurav/Documents/female_zero/s1.wav")

n_samples = int(fs*0.03)

data = data[:n_samples]
time_axis = np.array(range(n_samples))/fs

data_fft = fftpack.rfft(data)
data_fft = np.absolute(data_fft)
data_cep = np.log10(data_fft)
data_cep = fftpack.irfft(data_cep)

#Plot signal, cepstrum and spectrum vs time
plt.figure(1)
first = plt.subplot(311)
first.plot(time_axis, data)
first.set_title("Signal")
second = plt.subplot(312)
second.plot(time_axis, data_cep)
second.set_title("Cepstrum")
third = plt.subplot(313)
freqs = fftpack.rfftfreq(data.size, time_axis[1]-time_axis[0])
third.plot(freqs, 20*np.log10(data_fft))
third.set_title("Log magnitude spectrum")
real_ceps_pitch=data_cep(int(fs*2/1000):data_cep.size)
max1=real_ceps_pitch.max
for uu in range(real_ceps_pitch.size):
    if real_ceps_pitch(uu)==max1:
        sample_no=uu;
pitch_period_To=(int(fs*2/1000)+sample_no)*(1/fs)
pitch_freq_FO=1/pitch_period_To

#Plot autocorrelation
#out1 = np.correlate(data, data, mode='same')
#out2 = np.correlate(data, data, mode='full')
#plt.figure(2)
#outf = plt.subplot(211)
#outf.set_title("Same")
#outf.plot(out1)
#outs = plt.subplot(212)
#outs.plot(out2)
#outs.set_title("Full")

plt.show()
