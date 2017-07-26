#! /usr/bin/env python

import numpy as np
import sys
import os

from rnnlm import RNNLM
from utilities import parse_params
import argparse

from matplotlib import pyplot as plt
from scipy.stats import  pearsonr as pearson




#from matplotlib import pyplot as plt
from scipy.stats import  pearsonr as pearson
from sklearn.metrics import mean_squared_error as MSE

commandLineParser = argparse.ArgumentParser (description = 'Compute features from labels.')
commandLineParser.add_argument ('--seed', type=int, default = 100,
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('--name', type=str, default = 'rnnlm',
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('--load_path', type=str, default = './',
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('--debug', type=int, default = 0,
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('data', type=str,
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('input_wlist', type=str,
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('output_wlist', type=str,
                                help = 'Specify path to model which should be loaded')
commandLineParser.add_argument ('gscale', type=float,
                                help = 'Specify path to model which should be loaded')



def _create_dict(path, index):
        dict = {}
        path = os.path.join(path, index)
        with open(path, 'r') as f:
          for line in f.readlines():
                  line = line.replace('\n', '').split()
                  dict[line[1]]= int(line[0])+1
        return dict

def _word_to_id(data, path, index):
        vocab = _create_dict(path, index)
        return [[vocab[word] if vocab.has_key(word) else 0 for word in line] for line in data]


def process_data_rescore(data, path, input_index, output_index, bptt, spId=False):
  data_path = os.path.join(path, data)
  text=[]
  with open(data_path, 'r') as f:
    data = []
    slens=[]
    meta=[]
    for line in f.readlines():
      line = line.replace('\n', '').split()
      meta.append(line[:3])
      text.append(' '.join(line[3:]))
      if spId:
        line = line[1:]
      data.append(line)
      slens.append( len(line) -1 )
  in_data = _word_to_id(data, path, input_index)
  out_data =_word_to_id(data, path, output_index)


  if bptt==None:
    slens = np.asarray(slens, dtype=np.int32)
    input_processed_data = np.zeros((len(slens), np.max(slens)), dtype=np.int32)
    target_processed_data = np.zeros((len(slens), np.max(slens)), dtype=np.int32)

    for i in xrange(len(in_data)):
      input = in_data[i][:-1]
      output = out_data[i][1:]
      input_processed_data[i][0:slens[i]] = input
      target_processed_data[i][0:slens[i]] = output
    return target_processed_data, input_processed_data, slens, text, meta
  else:
    sequence_lengths = []
    for s in slens:
      if s <= bptt:
        sequence_lengths.append(s)
      else:
        lines = int(np.floor(s/float(bptt)))
        lens = [bptt]*lines
        if len(lens) > 0: sequence_lengths.extend(lens)
        s = s % bptt
        if s > 0:
          sequence_lengths.append(s)
    sequence_lengths = np.asarray(sequence_lengths, dtype=np.int32)
    #print np.mean(sequence_lengths), np.std(sequence_lengths),

    #print sequence_lengths.shape[0], len(id_data)
    input_processed_data = np.zeros((len(sequence_lengths), bptt), dtype=np.int32)
    target_processed_data = np.zeros((len(sequence_lengths), bptt), dtype=np.int32)
    row = 0
    for i, length in zip(xrange(len(in_data)), slens):
      input = in_data[i][:-1]
      output = out_data[i][1:]
      lines = int(np.ceil(length/float(bptt)))
      for j in xrange(lines):
        input_processed_data[row+j][0:sequence_lengths[row+j]] = input[j*bptt:(j+1)*bptt]
        target_processed_data[row+j][0:sequence_lengths[row+j]] = output[j*bptt:(j+1)*bptt]
      row+=lines

    return target_processed_data, input_processed_data, sequence_lengths, text, meta


def main(argv=None):
  args = commandLineParser.parse_args()
  if os.path.isdir('CMDs'):
    with open('CMDs/step_test_rnnlm.txt', 'a') as f:
      f.write(' '.join(sys.argv)+'\n')
  else:
    os.mkdir('CMDs')
    with open('CMDs/step_test_rnnlm.txt', 'a') as f:
      f.write(' '.join(sys.argv)+'\n')


  valid_data = process_data_rescore(args.data, path="data", spId=False, input_index=args.input_wlist, output_index=args.output_wlist, bptt=None)
  print len(valid_data)

  network_architecture = parse_params('./config')

  rnnlm = RNNLM(network_architecture=network_architecture,
                    seed=args.seed,
                    name=args.name,
                    dir='./',
                    load_path=args.load_path,
                    debug_mode=args.debug)

  scores=[]
  print valid_data[4]
  probabilities=rnnlm.rescore(valid_data)
  for probs, length, i in zip(probabilities, valid_data[2], xrange(len(probabilities))):
    print i
    valid_data[4][i][1] = str(np.sum(np.log(probs[:length])))
    scores.append(float(valid_data[4][i][0])+args.gscale*float(valid_data[4][i][1]))
    #print '\n'+'\n'.join([str(prob) for prob in np.log(probs[:length])])+'\n'

  meta=[' '.join(score) for score in valid_data[4]]
  text = [' '.join([met,txt]) for met, txt in zip(meta, valid_data[3])]
  data=sorted(zip(scores, text), reverse=True)
  score, text = zip(*data)
  for txt in text:
	print txt
if __name__ == '__main__':
  main()
