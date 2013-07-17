import numpy as np
import os
import sys
import struct
import matplotlib.pyplot as plt
from pylab import *

'''
REFERENCES:
	www.practicalcryptography.com - MFCC Guide
	'''

rate = 44100

def plot_wf(wf):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(range(len(wf)),wf)
	plt.show()

def read_frame(n,src=sys.stdin.fileno()):
	s = ''
	while len(s) < 4*n:
		s += os.read(sys.stdin.fileno(),4*n-len(s))
	return struct.unpack('f'*n, s)

def write_frame(frame,dst=sys.stdout.fileno()):
	os.write(dst,struct.pack('f'*len(frame),*frame))

def window(frame):
	w = np.hanning(len(frame))
	return np.multiply(w,frame)

def log_power_spectrum(frame):
	return np.log(np.abs(np.fft.rfft(frame)))

def split_frame(frame, nsamp, overlap):
	i = 0
	while True:
		head = i*(nsamp - overlap)
		if (head+nsamp)>len(frame):
			return # truncate last subframe
		yield frame[head:head+nsamp]
		i += 1

def freq_to_mels(freq):
	return 1125 * np.log(1 + freq/700.)

def mels_to_freq(mels):
	return 700 * (np.exp(mels/1125.) - 1)

def get_spectrum_x_axis(frame_len):
	return [ i * rate / float(frame_len) for i in range(frame_len)]

def make_mel_filterbank_funcs(fmin,fmax,size):
	melmin = freq_to_mels(fmin)
	melmax = freq_to_mels(fmax)
	melpts = np.linspace(melmin,melmax,size+2)
	peakfreqs = mels_to_freq(melpts)
	filter_funcs = []
	for i in range(size):
		def filter_func(freq,i=i):
			''' triangular mel filter '''
			if freq < peakfreqs[i]:
				return 0
			elif freq < peakfreqs[i+1]:
				return (freq - peakfreqs[i])/(peakfreqs[i+1]-peakfreqs[i])
			elif freq < peakfreqs[i+2]:
				return (peakfreqs[i+2]-freq)/(peakfreqs[i+2]-peakfreqs[i+1])
			else:
				return 0
		filter_funcs += [filter_func]
	return filter_funcs

def render_filterbank(filter_funcs, X):
	''' X = list of x-axis values (frequencies) at which
	  to evaludate the filter functions '''
	return [ [ filter_func(x) for x in X ] for filter_func in filter_funcs]

def filterbank_energies(filterbank,powerspec):
	fbe = []
	for f in filterbank:
		e = np.sum(np.multiply(f,powerspec))
		fbe += [e]
	return fbe


#raw = read_frame(int(0.5*rate))
raw = read_frame(2*rate)

fbf = make_mel_filterbank_funcs(400,10000,26)
fb = None

ps = []
fbes = []

for subframe in split_frame(raw, int(.025*rate), int(.015*rate)):
	w = window(subframe)
	write_frame(w)

	p = log_power_spectrum(w)
	ps += [p] #[50:200]] # save for plotting

	if not fb:
		fb = render_filterbank(fbf, get_spectrum_x_axis(len(p)))
		'''
		for f in fb:
			plot(f)
			show()
			'''
	
	fbe = filterbank_energies(fb,p)
	#print >> sys.stderr, "FBE=", fbe
	fbes += [fbe]




ps = np.transpose(ps)
ax1 = subplot(211)
imshow(ps,origin="lower")
subplot(212,sharex=ax1)
imshow(np.transpose(fbes),origin="lower")
#yticks(range(ps.shape[0]), get_spectrum_x_axis(ps.shape[1]))
show()
