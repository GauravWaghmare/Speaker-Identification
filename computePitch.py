import scipy.io.wavfile   #This library is used for reading the .wav file
from matplotlib.pyplot import plot, show, grid, figure, subplot
from numpy import amax, absolute, arange
import matplotlib.pyplot as plt
import numpy as np
import math
from spectrum.window import Window
import scipy
from scipy import fftpack
from scipy import signal


[Fs, y ]= scipy.io.wavfile.read('/home/manvi/Desktop/voicebiometric/10recordings/1.wav');
data = y

pitch_freq1=0.0
o = 1.0
Frame_size = 30
# pitch_freq = 0.0   ### array
Frame_shift = 10
window_period = Frame_size/1000.0
shift_period = Frame_shift/1000.0

print window_period
print shift_period

window_length = window_period*Fs
sample_shift = shift_period*Fs

print window_length
print sample_shift


Channel_2 = y
max_val = amax(absolute(Channel_2))
Channel_2 = np.asfarray(Channel_2, dtype= 'float')
y = Channel_2/float(max_val)
t = arange(1.0/Fs, (len(y)*1.0/Fs)+(1.0/Fs) , 1.0/Fs)


sum1=0.0
energy=0.0
autocorrelation=0.0

rangei = (math.floor((len(y))/sample_shift)- math.ceil(window_length/sample_shift))
print rangei
print int(rangei)


pitch_freq = np.zeros(shape=(rangei,))

for i in range(1, int(rangei+1)):
    # print i
    k = 1
    jlow = ((i-1)*sample_shift)+1
    jup = (((i-1)*sample_shift)+window_length)
    yy = np.zeros(shape=(jup,))

    # print jlow
    # print jup

    for j in range(int(jlow), int(jup)):
        print j
        yy[k] = y[j]
        k = k + 1
    print yy
    t = arange( 1.0/Fs, len(yy)*1.0/Fs, 1.0/Fs)
    # t=1/Fs:1/Fs:(length(yy)/Fs);
    t = t[1:len(t)/2]
    t *= 1000
    # t=(t(1:length(t)/2))*1000;

    dfty = absolute(fftpack.rfft(yy))
    dfty1 = dfty[1:dfty.size/2]
    tt = arange(1.0/Fs, len(dfty1), Fs)
    # tt = linspace(1/Fs,Fs,length(dfty1))
    for i in range(1, len(dfty)):
        if (dfty[i]==0):
            dfty[i]= 0.000000000001
    # end
    dftylog = np.log10(dfty)
    dftylog1 = dftylog[1:dftylog.size/2]
    yy = 10*dftylog1


    # //xtitle('Log Magnitude Spectrum','frequency in Hz');
    real_ceps=absolute(fftpack.irfft(dftylog));
    real_ceps = real_ceps[1:len(real_ceps)*1.0/2]
    # //real_ceps=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 5 4 3 2 1];
    real_ceps_pitch = real_ceps[16: len(real_ceps)]

    max1 = amax(real_ceps_pitch)
    # //max1=0;
    for uu in xrange(1, len(real_ceps_pitch)):
        if real_ceps_pitch[uu]==max1:
          sample_no=uu

    pitch_freq1 = 1.0/((Fs*2.0/1000+sample_no)*(1.0/Fs));
    pitch_freq[o] = pitch_freq1
    o=o+1
# end of uppermost for loop


kk = arange(1.0/Fs ,len(pitch_freq)*1.0*shift_period, shift_period)
    ##kk=1/Fs:shift_period:(length(pitch_freq)*shift_period)
    # subplot(4,1,3)
plot(kk,pitch_freq)
# xtitle('Pitch Contour obtained by cepstrum pitch estimation method')
