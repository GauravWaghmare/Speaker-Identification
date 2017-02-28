
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
o=1.0
Frame_size = 30.0
pitch_freq = 0.0
Frame_shift = 10.0
window_period = Frame_size/1000.0
shift_period = Frame_shift/1000.0
window_length = window_period*Fs
sample_shift = shift_period*Fs

Channel_2 = y
max_val = amax(absolute(Channel_2))
Channel_2 = np.asfarray(Channel_2, dtype= 'float')
y = Channel_2/float(max_val)
t = arange(1.0/Fs, (len(y)*1.0/Fs)+(1.0/Fs) , 1.0/Fs)


sum1=0.0
energy=0.0
autocorrelation=0.0


for i in xrange(q, (floor((length(y))/sample_shift)-ceil(window_length/sample_shift))):
    k = 1
    yy = 0
    for j in range((((i-1)*sample_shift)+1), (((i-1)*sample_shift)+window_length)):
        yy[k] = y[j]
        k = k + 1
    t = arrange( 1.0/Fs, len(yy)*1.0/Fs, 1.0/Fs)
    # t=1/Fs:1/Fs:(length(yy)/Fs);
    t = t[1:len(t)/2]
    t *= 1000
    # t=(t(1:length(t)/2))*1000;

    dfty = absolute(fftpack.rfft(yy))
    tt = arrange(1.0/Fs, dfty.size, Fs)
    # tt = linspace(1/Fs,Fs,length(dfty1))
    for i in range(1, len(dfty)):
        if (dfty[i]==0):
            dfty(i)=1D-16
    # end
    dftylog = np.log10(dfty)
    dftylog1 = dftylog[1:dftylog.size/2]
    yy = 10*dftylog1


//xtitle('Log Magnitude Spectrum','frequency in Hz');
real_ceps=absolute(fftpack.irfft(dftylog));
//real_ceps=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 5 4 3 2 1];
real_ceps_pitch=real_ceps[Fs*2/1000:]
max1=real_ceps_pitch.max()
//max1=0;
for uu in range(real_ceps_pitch.size)
    if real_ceps_pitch[uu]==max1:
      //max1=real_ceps_pitch(uu);
      sample_no=uu;
  pitch_freq1=1/((Fs*2/1000+sample_no)*(1/Fs));
  pitch_freq[o] = pitch_freq1
  o=o+1
  //pitch_freq(i)=1/(16+sample_no);
kk = arange(1.0/Fs ,len(pitch_freq)*1.0*shift_period, shift_period)
##kk=1/Fs:shift_period:(length(pitch_freq)*shift_period)
subplot(4,1,3)
plot(kk,pitch_freq,'.')
xtitle('Pitch Contour obtained by cepstrum pitch estimation method')


