import numpy as np
import crossValidation as cv

def average_distance(x_rdd,y_rdd):
    """ Compute the average distance between images in RDDs x and y
        where x and y contain key-value pairs (idx,img)

        inputs:
            x_rdd -> image RDD
            y_rdd -> image RDD

        outputs:
            d -> average distance between images
    """
    joined = x_rdd.join(y_rdd)
    return cv.total_rss(joined)**2

def ISNR(d_orig_deg,d_orig_res):
    """ Compute the increased signal to noise ratio between the
        and restored images.

        inputs:
            d_orig_deg -> average distance between original and 
                          degraded images
            d_orig_res -> average distance between original and restored images
    """
    return 20.0*np.log10(1.0*d_orig_deg/d_orig_res)
