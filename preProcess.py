import numpy as np
import os
import gzip
import pickle
import cv2
def import_images(datapath="../data/mnist",MNIST_TRAIN = 'mnist.pkl'):
    full_path = os.path.join(datapath,MNIST_TRAIN)
    with open(full_path,'rb') as f:
        mnist = pickle.load(f)["training_images"]
    print("Done Loading")
    mnist = np.reshape(mnist,[-1,28,28])
    return mnist
def pad_and_resize():
    pass
def synthetic_degredation():
    pass
def save_image(img):
    cv2.imwrite("img_0.jpg",img)
img = import_images()[0]
save_image(img)
