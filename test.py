import numpy as np
import os
import sys
import struct
import matplotlib.pyplot as plt
from pylab import *

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

raw = read_frame(int(0.5*rate))
raw = read_frame(1*rate)

ps = []

for subframe in split_frame(raw, int(.025*rate), int(.015*rate)):
	w = window(subframe)
	write_frame(w)
	
	ps += [log_power_spectrum(w)[0:200]]

ps = np.transpose(ps)
imshow(ps,origin="lower")
show()
