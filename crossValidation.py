import numpy as np
import imageProcessing as ip
from tqdm import tqdm

def rss(x,xhat):
    """ Compute the residual sum of squares (RSS) for a single image.

        inputs:
            x    -> clean training image
            xhat -> restored image
        
        outputs:
            rss -> sum_i(||y_i - x_i||^2) where sum is over each pixel i 
                   of the images
    """
    return np.sum(np.add(x,-1.0*xhat)**2)

def total_rss(xhat_rdd):
    """ Compute the total RSS between two RDDs of images.

        inputs:
            xhat_rdd -> rdd of key value pairs of the form (idx,(x,xhat)) 
                        where x is the clean image and xhat is the restored

        outputs:
            total_rss -> total RSS between clean and restored images
    """    
    rss_rdd = xhat_rdd.mapValues(lambda (x,xhat): rss(x,xhat))
    return rss_rdd.values() \
                  .reduce(lambda rss1,rss2: rss1+rss2) / rss_rdd.count()

def cross_validate(joined_rdd,k=10,lam=[0]):
    """ Perform cross validation between clean and restored image RDDs.

        inputs:
            joined_rdd -> joined key-value pair rdd of clean and degraded images
            k     -> number of folds (default = 10)
            lam   -> list of lambda values to test

        outputs:
            rss_list -> list of total_rss computed for each value of lam
    """
    N = joined_rdd.count()              # total number of samples

    # joined_rdd = x_rdd.join(y_rdd) # join image pairs
     
    rss = []
    for j in tqdm(range(len(lam))):
        rssk = []
        for i in tqdm(range(k)):
            # split data into training and validation sets
            train_rdd = joined_rdd.filter(lambda (idx,val):  
                                          index_to_fold(idx,N,k) != i).cache()
            validate_rdd = joined_rdd.filter(lambda (idx,val):  
                                             index_to_fold(idx,N,k) == i).cache()

            # learn restoration filter from training set
            h = ip.estimate_filter_full(train_rdd)

            # restore validation images
            restore_fun = lambda y: ip.restore_image(y,h,lam[j])
            xhat_rdd = validate_rdd.mapValues(lambda (x,y): (x,restore_fun(y)))

            # compute RSS 
            rssk.append(total_rss(xhat_rdd))

        # average rss over k folds
        rss.append(np.mean(rssk))
    
    # return list of rss values (one for each lambda)
    return rss

def index_to_fold(idx,N,k):
    """ Maps an index to a fold label.

        inputs:
            idx -> data point index
            N   -> number of data points
            k   -> number of folds

        outputs:
            fold -> fold label (0 to k-1)
    """
    N_per_fold = np.floor(1.0*N/k)
    return np.floor(idx/N_per_fold)
