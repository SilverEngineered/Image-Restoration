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
            numpy array (60000,28,28) where each value is a 8 bit
            integer representing the intensity of the grayscale pixel
    """
    full_path = os.path.join(datapath,MNIST_filename)
    with open(full_path,'rb') as f:
        mnist = pickle.load(f)["training_images"]
    print("Done Loading")
    mnist = np.reshape(mnist,[-1,28,28])
    return mnist
def scale_images(images,scaling=[0,1],dtype=np.float32):
    """  Scale an array of images to the specified scaling range

         inputs:
             images -> numpy array of images in the form (x,i,j)
             where x is the number of images
                   i is the number of rows
                   j is the number of columns

             scaling ->  integer tuple (min,max)
             where min is the minimum value after scaling
                   max is the maximum value after scaling
    
         outputs:
             numpy array of images scaled from min to max
    """
    min_data, max_data = [float(np.min(images)), float(np.max(images))]
    min_scale, max_scale = [float(scaling[0]), float(scaling[1])]
    data = ((max_scale - min_scale) * (images - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)
def pad_images(images,pad_shape=(0,0)):
    """  Pads images with 0s to fit a desired shape

         inputs:
             images -> numpy array of images in the form (x,i,j)
             where x is the number of images
                   i is the number of rows
                   j is the number of columns
            
             padding -> integer tuple (row,col)
             where row is the number of 0s to pad to the beggining and end of each row
                   col is the number of 0s to pad to the beggining and end of each col

         outputs:
             padded images of size (x,i + 2 * row,j + 2 * col) 
    """
    padded_images = []
    for i in range(images.shape[0]):
        padded_images.append[np.pad(images,pad_shape,'constant')]
    return np.array(padded_images)
def synthetic_degredation(images,deg_function):
    """  Applies a degredation to all images in image_array
         by applying a degredation function
        inputs:
            images  -> numpy array of images in the form (x,i,j)  
            where x is the number of images
            i is the number of rows
            j is the number of columns

            deg_function -> (image -> image) function
         
        outputs:
            degredated images -> (x,i,j)
    """
    return np.array(list(list(images).map(lambda x: deg_function(x))))
def save_image(img,img_path="img_0.jpg"):
    """  Save output image to img_path

    """
    cv2.imwrite(img_path,img)

#img = import_images()[0]
#save_image(img)
