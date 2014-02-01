import numpy as np
import os
import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from pylab import *
from scipy import fftpack
from collections import defaultdict
import random
import json
import mlpy


def plot_wf(wf):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(range(len(wf)),wf)
  plt.show()

def normalize(v):
  vmax = np.max(v)
  vmin = np.min(v)
  vmag = vmax - vmin
  if (vmag == 0):
    return np.empty(np.size(v))
  vnorm = (np.array(v) - vmin) * 1/vmag
  return vnorm

def plot_feats(feats):
  for feat in feats:
    plot(feats[feat], label=feat)
  legend()

def truncate_feats_set(feats_set):
  ''' truncate each feature time series to whatever is the
   shortest '''
  # deterimine shortest timeseries
  minlen = None
  for feats in feats_set:
    for feat in feats:
      if not minlen:
        minlen = len(feats[feat])
      else:
        minlen = min(minlen,len(feats[feat]))
  # truncate
  for feats in feats_set:
    for feat in feats:
      feats[feat] = feats[feat][:minlen]
  # return truncated size
  return minlen



if __name__=="__main__":

  realfeats = json.load(open('scots.feat'))
  realmelgram = json.load(open('scots.melgram'))
  compfeats = json.load(open('comp.feat'))
  compmelgram = json.load(open('comp.melgram'))
  compttsalign = [json.loads(line) for line in open('comp.ttsalign').readlines()]
  transcript = open('transcript').read()
  frame_offset_ms = 0.01 * 1000  # ms per frame

  '''for feats in [realfeats, compfeats]:
    del feats['power']
    del feats['hi-lo']'''

  print "FEATURES: ", compfeats.keys()

  # truncate to shortest feature
  #truncate_feats_set([realfeats,compfeats])

  x = realfeats['high']
  y = compfeats['high']
  dist, cost, path = mlpy.dtw_std(x,y,dist_only=False,squared=False)
  print "Dist = ", dist

  fig = plt.figure(1)
  ax1 = fig.add_subplot(311)
  plot1 = plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
  plot2 = plt.plot(path[0], path[1], 'w')
  xlim = ax1.set_xlim((-.5, cost.shape[0]-0.5))
  ylim = ax1.set_ylim((-.5, cost.shape[1]-0.5))

  ax2 = fig.add_subplot(312)
  yposs = [10,20]
  yposi = 0
  for event in compttsalign:
    if event['type'] == 'WORD':
      frame_num = event['audio_position']/frame_offset_ms
      word = transcript[event['text_pos']-1:event['text_pos']-1+event['length']]
      yposi += 1
      ypos = yposs[yposi%len(yposs)]
      plt.text(frame_num, ypos, word, fontsize=10, weight='bold')
      print "FRAMENUM=%d, WORD=%s, YPOS=%d" % (frame_num, word,ypos)
  plot3 = plt.imshow(np.transpose(compmelgram), origin='lower')

  ax2 = fig.add_subplot(313)
  yposs = [10,20]
  yposi = 0
  for event in compttsalign:
    if event['type'] == 'WORD':
      comp_frame_num = int(event['audio_position']/frame_offset_ms)

      for i in range(len(path[0])):
        if path[1][i] > comp_frame_num:
          real_frame_num = path[0][i-1]
          break

      word = transcript[event['text_pos']-1:event['text_pos']-1+event['length']]
      yposi += 1
      ypos = yposs[yposi%len(yposs)]
      plt.text(real_frame_num, ypos, word, fontsize=10, weight='bold')
      print "CFN/RFN=(%d,%d) WORD=%s, YPOS=%d" % (comp_frame_num, real_frame_num, word, ypos)
  plot3 = plt.imshow(np.transpose(realmelgram), origin='lower')



  plt.show()

'''
  fig = plt.figure()
  ax1 = fig.add_subplot(211)
  plot_feats(realfeats)
  ax2 = fig.add_subplot(212)
  plot_feats(compfeats)
  show()'''


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
