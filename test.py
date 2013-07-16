import numpy as np
import os
import sys
import struct
import matplotlib.pyplot as plt


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

#read_frame(4000)
raw = read_frame(8000)
wd = window(raw)
####plot_wf(raw)
#plot_wf(wd)

write_frame(raw)
write_frame(wd)

while True:
	raw = read_frame(8000)
	write_frame(window(raw))
