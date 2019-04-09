import numpy as np
import os
import gzip
import pickle
import cv2
def import_images(datapath="../data/mnist",MNIST_filename = 'mnist.pkl'):
    """  Import pickled mnist data and save it as a numpy array in the
         shape of (60000,28,28) where each 0th element of the array
         is a black and white MNIST image

         inputs:
             datapath -> A string representing the datapath to the
                         pickled mnist file
             MNIST_filename -> A string representing the name of the
                               pickled mnist file
         outputs:
            Numpy Array (60000,28,28)
    """
    full_path = os.path.join(datapath,MNIST_filename)
    with open(full_path,'rb') as f:
        mnist = pickle.load(f)["training_images"]
    print("Done Loading")
    mnist = np.reshape(mnist,[-1,28,28])
    return mnist
def pad_and_resize(images,padding=9):
    pass
def synthetic_degredation():
    pass
def save_image(img):
    cv2.imwrite("img_0.jpg",img)
img = import_images()[0]
save_image(img)
