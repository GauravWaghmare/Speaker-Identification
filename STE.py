import scipy.io.wavfile   #This library is used for reading the .wav file
from matplotlib.pyplot import plot, show, grid, figure, subplot
from numpy import amax, absolute, arange
import matplotlib.pyplot as plt
import numpy as np
import math
from spectrum.window import Window
import scipy

# -------------------------------------------------------
# to create a hamming window
# use numpy.hamming(nuber of points in output window)
# -------------------------------------------------------

def plot_wave(t, y, xlabel, ylabel):
	plot(t,y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)


def get_original_wave(speech, sampling_freq):
	y = speech
	ts = 1.0/sampling_freq
	tot_duration = (1.0/sampling_freq)*len(y)
	t = arange(1.0/sampling_freq,(tot_duration+ts),1.0/sampling_freq)
	return (t,y)


def compute_zerocrossing(speech,  sample_shift, window_len):
	y = speech.copy()
	win = np.hamming(window_len)
	sum1 = 0
	num_of_frames = (math.floor(len(y)/sample_shift) - math.ceil(window_len/sample_shift))
	print num_of_frames
	zerocrossing = np.zeros(shape=(num_of_frames,))
	jj = 1
	i = 0
	while i < num_of_frames:
		y[((i-1)*sample_shift)+1] = y[((i-1)*sample_shift)+1]*win[jj]
		jj += 1
		j = (((i-1)*sample_shift)+2)
		while j< (((i-1)*sample_shift) + window_len):
			y[j] = y[j]*win[jj]
			jj+=1
			yy = y[j]*y[j-1]
			if yy<0:
				sum1 +=1
			j += 1
		zerocrossing[i] = sum1*1.0/(2*window_len)
		sum1 = 0
		jj = 1
		i += 1
	w = 0
	print len(zerocrossing)
	return zerocrossing


def short_term_energy(speech, sample_shift, window_len):
	y = speech.copy()
	win = np.hamming(window_len)
	sum1 = 0
	num_of_frames = (math.floor(len(y)/sample_shift) - math.ceil(window_len/sample_shift))
	print num_of_frames
	energy = np.zeros(shape=(num_of_frames,))
	jj = 1
	i = 0
	while i < num_of_frames:
		j = (((i-1)*sample_shift)+1)
		while j< (((i-1)*sample_shift) + window_len):
			y[j] = y[j]*win[jj]
			jj+=1
			yy = y[j]*y[j]
			sum1 = sum1 + yy
			j += 1
		energy[i] = sum1
		sum1 = 0
		jj = 1
		i += 1
	w = 0
	# c = energy
	print len(energy)
	return energy



def fft(speech, frame_shift):
	y = speech.copy()
	print len(y)
	power_spectra = np.fft.rfft(y)
	return power_spectra


file_name = '/home/manvi/Desktop/voicebiometric/10recordings/1.wav'

sampling_freq, signal = scipy.io.wavfile.read(file_name)
print sampling_freq


bl, al = scipy.signal.iirfilter(15, 4000.0*2.0/sampling_freq, rp=1, rs=100, btype='lowpass', analog=False, ftype='ellip')
bl2, al2 = scipy.signal.iirfilter(15, 200.0*2.0/sampling_freq, rp=1, rs=100, btype='lowpass', analog=False, ftype='ellip')
bh, ah = scipy.signal.iirfilter(15, 4000.0*2.0/sampling_freq, rp=1, rs=100, btype='highpass', analog=False, ftype='ellip')
wl, hl = scipy.signal.freqz(bl, al)
wh, hh = scipy.signal.freqz(bh, ah)

# figure(1)
# plt.title('Digital filter frequency response')
# plt.plot(wl, 20*np.log10(np.abs(hl)))
# plt.title('Digital filter frequency response')
# plt.ylabel('Amplitude Response [dB]')
# plt.xlabel('Frequency (rad/sample)')
# plt.grid()

# figure(2)
# plt.title('Digital filter frequency response')
# plt.plot(wh, 20*np.log10(np.abs(hh)))
# plt.title('Digital filter frequency response')
# plt.ylabel('Amplitude Response [dB]')
# plt.xlabel('Frequency (rad/sample)')
# plt.grid()
# show()


frame_size = 5.0 # in miliseconds
frame_shift = 2.5 # in miliseconds

frame_size = frame_size/1000.0 # to convert seconds to miliseconds
frame_shift = frame_shift/1000.0  # to convert seconds to miliseconds
window_len = frame_size*sampling_freq # Number of samples in frame_size
sample_shift = frame_shift*sampling_freq # Number of samples shifted


# Channel_1 = signal[:,0]
# Channel_2 = signal[:,1]
Channel_2 = signal
x = absolute(Channel_2)
max_val = amax(x)
Channel_2 = np.asfarray(Channel_2, dtype= 'float')
signal = Channel_2/float(max_val)


t, y = get_original_wave(signal, sampling_freq)
print len(t)
print len(y)
print sampling_freq
figure(1)
plot_wave(t, y, 'time in seconds', 'wave amp')


# fft_signal = fft(signal, window_len)
# power_spectra = np.square(fft_signal)



# # Channel_2 = signal
# filtered_data_l = scipy.signal.lfilter(bl, al, signal)
# filtered_data_h = scipy.signal.lfilter(bh, ah, signal)
# print len(filtered_data_l)
# print len(t)
# # figure(2)
# # plot_wave(t, filtered_data_l, 'time in seconds', 'filtered_data_l')

# # figure(3)
# # plot_wave(t, filtered_data_h, 'time in seconds', 'filtered_data_h')



energy_signal = short_term_energy(signal, sample_shift, window_len)
# energy_fdh = short_term_energy(filtered_data_h, sample_shift, window_len)
# energy_fdl = short_term_energy(filtered_data_l, sample_shift, window_len)
# print len(energy_fdl)
# print len(energy_fdh)

tt = arange(1.0/sampling_freq, (1.0*len(energy_signal)*frame_shift) ,frame_shift)
print len(tt)
print len(energy_signal)
figure(4)
plot_wave(tt,energy_signal, 'time in seconds', 'signal energy')
show()
# figure(5)
# plot_wave(tt,energy_fdl, 'time in seconds', 'lowwpass energy')
# figure(6)
# plot_wave(tt,energy_fdh, 'time in seconds', 'highpass energy')

# filtered_signal = signal - filtered_data_h
# ff_name = file_name[:-4] + "PPD.wav"
# scipy.io.wavfile.write(ff_name, sampling_freq, filtered_signal)

# SNR = np.divide(energy_fdl, energy_fdh)
# figure(7)
# grid()
# plot_wave(tt, SNR, 'time', 'SNR')
# figure(8)
# plot_wave(tt, 20*np.log10(SNR), 'time', 'SNR')
# grid()
# # show()


# zerocrossing = compute_zerocrossing(signal, sample_shift, window_len)
# print len(zerocrossing)
# # tt = arange(1.0/sampling_freq, (1.0*len(zerocrossing)*frame_shift) ,frame_shift)
# # print len(tt)
# figure(9)
# grid()
# plot_wave(tt, zerocrossing, 'time in seconds', 'zerocrossing')
# show()

# # show()