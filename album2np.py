import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import matplotlib.pyplot as plt 
import os, time, itertools, pickle, random, glob, imageio
from PIL import Image

img_size = 128
datapath = os.path.join('WGAN-GP-TensorFlow', 'datasets', 'album')
audiopath = os.path.join('audio','spec256')
samplename = 'x_samples_128.npy'
labelename = 'x_labels_128.npy'
listname = 'x_datalist_128.npy'

def dataset_load_album(datapath, labelpath, imgsize):
    """
    datapath 'jpeg' file of album image
    labelpath 'png' file of spectrogram 
    """
    datalist = os.listdir(datapath)
    datasize = len(datalist)
    #print(datasize)
    
    data_ = np.zeros((datasize, imgsize, imgsize, 3), dtype='float32')
    label_ = np.zeros((datasize, imgsize, imgsize, 3), dtype='float32')
    outputlist = []
    nonelist = []
    for i, fname in enumerate(datalist):
        if glob.glob(os.path.join(labelpath, fname[:-4] + '*')):
        #f os.path.isfile(os.path.join(labelpath, fname[:-4] + 'png')):
            #print('there is file')
            img_d = Image.open(os.path.join(datapath, fname)
                              ).resize((imgsize, imgsize))
            img_l = Image.open(os.path.join(labelpath, fname[:-4]+'png')
                              ).convert('RGB').resize((imgsize, imgsize))
            data_[i] = np.asarray(img_d)
            label_[i] = np.asarray(img_l)
            img_d.close()
            img_l.close()
            #print(fname[:-4])
            outputlist.append(fname[:-5])
            
        else:
            nonelist.append(i)
            #print(i, 'is empty')
            
    data = np.delete(data_, nonelist, 0)
    label = np.delete(label_, nonelist, 0)
    return data, label, outputlist



samples, labels, datalist= dataset_load_album(
    datapath, audiopath, img_size)

samples = samples/255
labels = labels/255

input_dim = img_size * img_size * 3
num_sample = samples.shape[0]

np.save(samplename, samples)
np.save(labelename, labels)
np.save(listname, datalist)
