import numpy as np
import os
import gzip
def import_images(datapath="./",MNIST_TRAIN = 'train-images-idx3-ubyte.gz'):
    f = gzip.open(MNIST_TRAIN,'r')
    x=f.read(16)
    print(x)
def pad_and_resize():
    pass
def synthetic_degredation():
    pass
import_images()
