import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
import math
import h5py
import glob

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def human_sort(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

def get_models(save_dir, model_name):
    models = sorted(glob.glob(save_dir+'/'+model_name+'*'), key=os.path.getmtime)
    if len(models)==0:
        print('[INFO] Not find model', model_name, ' under ', save_dir)
    models = [ os.path.splitext(m)[0] for m in models]
    models = list(set(models))    # removyye duplication
    models = human_sort(models)
    return models

def save_h5(h5file, data, label):
    # data shape = [bsize, c, h, w, d], label shape = [bsize, h, w, d]
    if len(data.shape) == 5 and len(label.shape)==4:
        assert data.shape[0] == label.shape[0] == 1
        assert data.shape[2:] == label.shape[1:]
        data = np.squeeze(data, axis=0)
        label = np.squeeze(label, axis=0)
    

    assert len(data.shape) == 4 and len(label.shape)== 3
    assert data.shape[1:] == label.shape[:]
    f=h5py.File(h5file,'w')
    f.create_dataset('data',data=data)
    f.create_dataset('label', data=label)


def readClassesHist(filename, num_classes):
    if Path(filename).is_file():
        print('Classes Histogram file:', filename)
        counts = list()
        with open(filename, 'r') as f:
            for index, line in enumerate(f):
                val = line.split()[-1]
                counts.append(float(val))
        counts = np.array(counts)
    else:
        counts = np.ones(num_classes)
        counts[0] = counts[-1] = 0
    print(counts)
    return counts


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                         count=int(width)*int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))

