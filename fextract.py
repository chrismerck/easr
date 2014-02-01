import numpy as np
import os
import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import *
from scipy import fftpack
from collections import defaultdict
import random
import json

'''
REFERENCES:
  www.practicalcryptography.com - MFCC Guide
  '''


def plot_wf(wf):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(range(len(wf)),wf)
  plt.show()

def read_frame(n=None,src=sys.stdin.fileno()):
  if not n:
    s = src.read()
    n = len(s)/4
  else:
    s = ''
    while len(s) < 4*n:
      s += src.read(4*n-len(s))
  return struct.unpack('f'*n, s)

def write_frame(frame,dst=sys.stdout.fileno()):
  os.write(dst,struct.pack('f'*len(frame),*frame))

def window(frame):
  w = np.hanning(len(frame))
  return np.multiply(w,frame)

def periodogram(frame):
  return (np.abs(np.fft.rfft(frame))**2)/float(len(frame))

_preemph_prev = 0
def preemph(in_wave,alpha=0.97):
  global _preemph_prev
  out_wave = np.empty(len(in_wave), 'd')
  out_wave[0] = in_wave[0] - alpha*_preemph_prev
  for i in range(1, len(in_wave)):
    out_wave[i] = in_wave[i] - alpha*out_wave[i-1]
  _preemph_prev = in_wave[-1]
  return out_wave

def add_noise(in_wave,alpha=0.001):
  out_wave = np.empty(len(in_wave), 'd')
  for i in range(len(in_wave)):
    out_wave[i] = in_wave[i] + random.random()*alpha
  return out_wave

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

def get_spectrum_x_axis(frame_len, rate):
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

def filterbank_log_energies(filterbank,powerspec):
  fbe = []
  for f in filterbank:
    e = np.log(np.sum(np.multiply(f,powerspec)))
    fbe += [e]
  return fbe

def normalize(v):
  vmax = np.max(v)
  vmin = np.min(v)
  vmag = vmax - vmin
  if (vmag == 0):
    return np.empty(np.size(v))
  vnorm = (np.array(v) - vmin) * 1/vmag
  return vnorm



#EXPORT
def extract_features(filename, rate=44100, n=None):
  f = open(filename,'r')

  raw = read_frame(n,src=f)
  raw = preemph(add_noise(raw))

  fbf = make_mel_filterbank_funcs(300,12000,32)
  fb = None

  ps = []
  fbes = []
  featsd = defaultdict(list)

  for subframe in split_frame(raw, int(.025*rate), int(.015*rate)):
    w = window(subframe)
    p = periodogram(w)

    if not fb:
      fb = render_filterbank(fbf, get_spectrum_x_axis(len(p),rate))
    
    fbe = filterbank_log_energies(fb,p)
    fbes += [fbe]
    featsd['power'] += [np.sum(fbe)]
    featsd['high'] += [np.sum(fbe[27:32])]
    featsd['low'] += [np.sum(fbe[0:8])]

  featsd = {x:normalize(featsd[x]) for x in featsd}

  featsd['hi-lo'] = normalize(featsd['high']-featsd['low'])

  return featsd, fbes # dictionary of features, mel spectrogram

def flatten_nparray(a):
  if type(a) == np.ndarray:
    return flatten_nparray(a.tolist())
  elif type(a) == list:
    return [ flatten_nparray(x) for x in a ]

if __name__=="__main__":

  filebase = sys.argv[1]
  rate = 44100
  featsd, melgram = extract_features(filebase + '.raw')#, n=2*rate)
  for feat in featsd:
    featsd[feat] = featsd[feat].tolist()

  #melgram = flatten_nparray(melgram)

  json.dump(featsd,open(filebase+'.feat','w'))
  json.dump(melgram,open(filebase+'.melgram','w'))


"""
  #jraw = read_frame(int(0.5*rate))
  raw = read_frame(int(2*rate))
  raw = preemph(add_noise(raw))
  write_frame(raw)


  fbf = make_mel_filterbank_funcs(300,12000,32)
  fb = None

  ps = []
  fbes = []
  featsd = defaultdict(list)

  for subframe in split_frame(raw, int(.025*rate), int(.015*rate)):
    w = window(subframe)
    #write_frame(w)

    p = periodogram(w)
    ps += [p] #[50:200]] # save for plotting

    if not fb:
      fb = render_filterbank(fbf, get_spectrum_x_axis(len(p),rate))
    
    fbe = filterbank_log_energies(fb,p)
    #print >> sys.stderr, "FBE=", fbe
    fbes += [fbe]
    #mfcc = fftpack.dct(fbe)[0:len(fbe)/2]
    #mfccs += [mfcc]
    featsd['power'] += [np.sum(fbe)]
    featsd['high'] += [np.sum(fbe[27:32])]
    featsd['low'] += [np.sum(fbe[0:8])]

  featsd = {x:normalize(featsd[x]) for x in featsd}

  featsd['hi-lo'] = normalize(featsd['high']-featsd['low'])

  ps = np.transpose(ps)
  ax1 = subplot(211)
  #imshow(ps,origin="lower")
  #imshow((np.transpose(mfccs)),origin="lower",interpolation="nearest")
  #colorbar()
  plot(featsd['low'])
  plot(featsd['hi-lo'])
  subplot(212,sharex=ax1)
  imshow(np.transpose(fbes),origin="lower")
  #yticks(range(ps.shape[0]), get_spectrum_x_axis(ps.shape[1]))
  show()

  '''
  anim_steps = 50

  def update_line(num, data, line):
    frames = len(mfccx)
    tmin = num*frames/(anim_steps)
    tmax = (num+1)*frames/(anim_steps)

    line.set_data(mfccx[tmin:tmax],mfccy[tmin:tmax])
    print >> sys.stderr, num
    return line,

  fig1 = plt.figure()
  data = np.random.rand(2,25)
  plt.xlim(np.min(mfccx),np.max(mfccx))
  plt.ylim(np.min(mfccy),np.max(mfccy))
  l, = plt.plot([],[],'r-')
  line_ani = animation.FuncAnimation(fig1, update_line, anim_steps, fargs=(data, l),
      interval=50,blit=True)

  #plot(mfccx,mfccy)
  show()
  '''
  """
