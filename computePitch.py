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


Fs, y= scipy.io.wavfile.read('/home/manvi/Desktop/voicebiometric/10recordings/1.wav');
data = y

pitch_freq1=0.0
o = 0
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

for i in range(0, int(rangei)):
    print "i = " + str(i)
    k = 0
    jlow = ((i-1)*sample_shift)+1
    jup = (((i-1)*sample_shift)+window_length)

    yy = np.empty(shape=(jup-jlow+1,))

    # print "len(yy) = " + str(len(yy))
    for j in range(int(jlow), int(jup)):
        # print j
        yy[k] = y[j]
        k = k + 1

    t = arange( 1.0/Fs, (len(yy)*1.0/Fs + (1.0/Fs)), 1.0/Fs)
    # print len(t)/2
    ## t=1/Fs:1/Fs:(length(yy)/Fs);
    
    t = t[:(len(t)/2)]
    t *= 1000
    # print len(t)
#     # t=(t(1:length(t)/2))*1000;
    # print dfty
    dfty = absolute(fftpack.fft(yy))
    # print dfty.shape
    # print dfty
    dfty1 = dfty[:(dfty.size/2)]
    tt = np.linspace(1.0/Fs, Fs, len(dfty1))
#     # tt = linspace(1/Fs,Fs,length(dfty1) )
    for i in range(0, len(dfty)):
        if (dfty[i]==0):
            dfty[i]= 1e-16
    # end
    dftylog = np.log10(dfty)
    # print dftylog
    dftylog1 = dftylog[:dftylog.size/2]
    # print dftylog1
    yy = 10*dftylog1
    # print yy


    # print len(yy)
#     # //xtitle('Log Magnitude Spectrum','frequency in Hz');
    real_ceps = absolute(np.fft.ifft(dftylog))
    # print real_ceps
    real_ceps = real_ceps[:(len(real_ceps)/2)]
    real_ceps_pitch = real_ceps[16: len(real_ceps)]

    # print real_ceps_pitch

    max1 = amax(real_ceps_pitch)
    # print "\t  " + str(max1)
    for uu in xrange(0, len(real_ceps_pitch)):
        if real_ceps_pitch[uu]==max1:
          sample_no=uu
          # print sample_no

    # print sample_no
    pitch_freq1 = 1.0/((Fs*2.0/1000+sample_no)*(1.0/Fs));
    print "\t " + str(pitch_freq1)
    # print o
    pitch_freq[o] = pitch_freq1
    o=o+1
# # end of uppermost for loop


kk = arange(1.0/Fs ,len(pitch_freq)*shift_period, shift_period)
    ##kk=1/Fs:shift_period:(length(pitch_freq)*shift_period)
    # subplot(4,1,3)
print len(kk)
print len(pitch_freq)
plot(kk,pitch_freq)
axes = plt.gca()
axes.set_ylim([0,1000])
# plt.axis([0 8 0 1000])
show()
# xtitle('Pitch Contour obtained by cepstrum pitch estimation method')
