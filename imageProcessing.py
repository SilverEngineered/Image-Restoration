import numpy as np
from scipy.stats import norm
from tqdm import tqdm
def degrade_image(x,sigma,N_filt,noise_power):
    """ Performs artificial degradation to a single image.
        The image is first blurred then Gaussian noise is added.
        
        inputs:
            x           -> input image np array (zero padded)
            sigma       -> parameter determining the size of the blur filter
            N_filt      -> size of square filter support
            noise_power -> variance of additive Gaussian noise
        outputs:
            y -> degraded output image
    """
    x_filt = filter_image(x,sigma,N_filt)
    return add_noise(x_filt,noise_power)
def degrade_images(x,sigma,N_filt,noise_power):
    """ Degreades multiple images """
    images = []
    for i in tqdm(x):
        images.append(degrade_image(i,sigma,N_filt,noise_power))
    return np.array(images)
def filter_image(x,sigma,N_filt,filt_type="gauss"):
    """ Applies linear filter.

        inputs:
            x         -> input image np array (zero padded)
            sigma     -> parameter determining the size of the blur filter
            Nfilt     -> size of square filter support
            filt_type -> filter shape (default=gauss)
        outputs:
            y -> filtered image real part only
    """
    H = gauss_filter(sigma,N_filt)
    N_im = x.shape
    N_pad1 = N_im[0]-N_filt
    N_pad2 = N_im[1]-N_filt
    Hpad = np.pad(H,((0,N_pad1),(0,N_pad2)),'constant')
    y = np.fft.ifft2(np.fft.fft2(Hpad)*np.fft.fft2(x))
    return np.real(y)
 
def inverse_filter_image(x,sigma,N_filt,filt_type="gauss",lam=0):
    """ Applies linear inverse filter with regularization.

        inputs:
            x         -> input image np array (zero padded)
            sigma     -> parameter determining the size of the blur filter
            N_filt    -> size of square filter support
            filt_type -> filter shape (default=gauss)
            lam       -> regularization parameter
        outputs:
            y -> filtered image
    """
    
    H = gauss_filter(sigma,N_filt)
    N_im = x.shape
    N_pad1 = N_im[0]-N_filt
    N_pad2 = N_im[1]-N_filt
    Hpad = np.pad(H,((0,N_pad1),(0,N_pad2)),'constant')
    Hfft = np.fft.fft2(Hpad)
    Hffti = 1.0 / (np.add(Hfft,1.0*lam))
    y = np.fft.ifft2(Hffti*np.fft.fft2(x))
    return np.real(y)

def restore_image(y,h,lam=0):
    """ Restores an image using a regularized inverse filter.

        inputs:
            y   -> image to restore
            h   -> filter impulse response (same size as x)
            lam -> regularization parameter
       
        outputs:
            xhat -> restored image
    """
    H = np.fft.fft2(h)
    Hmag = np.conj(H)*H
    Hi_denom = (1.0 + 0.0j)*Hmag + (1.0 + 0.0j)*lam
    Hi = np.divide(H,Hi_denom)
    Y = np.fft.fft2(y)
    Xhat = np.multiply(Y,Hi)
    return np.real(np.fft.ifft2(Xhat))

def estimate_filter(x,y):
    """ Given an input and noisy output, estimate the filter.

        inputs:
            x -> input image
            y -> noisy filtered output image

        outputs:
            h -> estimated filter impulse response
    """
    X = np.fft.fft2(x)
    Y = np.fft.fft2(y)
    H = (1.0 + 0.0j)*Y / ( (1.0 + 0.0j)*X )
    h = np.fft.ifft2(H)
    return np.real(h)

def estimate_filter_full(joined_rdd):
    """ Given an rdd of clean images and noisy outputs, estimate the filter 
        by averaging filter estimates of each image pair.

        inputs:
            joined_rdd -> key value pairs of the form (idx, (x,y)) where
                          x is the clean image and y is the noisy degraded image

        outputs:
            h -> estimated filter impulse response
    """
    h_estimates = joined_rdd.mapValues(lambda (x,y): estimate_filter(x,y))
    h = h_estimates.values() \
                   .reduce(lambda h1,h2: np.add(h1,h2)) / h_estimates.count()
    return h

def add_noise(x,noise_power):
    """ Adds white Gaussian noise to image.

        inputs:
            x           -> input image np array
            noise_power -> variance of additive noise
        outputs:
            y -> noisy image
    """
    size_x = x.shape
    noise = np.random.normal(0,np.sqrt(noise_power),size_x)
    return np.add(x,noise)

def gauss_filter(sigma,N):
    """ Returns Gaussian filter impulse response as np array.

        inputs:
            sigma -> parameter determining the size of the blur filter
            N     -> size of square filter support
        outputs:
            H -> Gaussian filter impulse response (filt_size-by-filt_size)
    """
    if N%2 == 0:
        raise ValueError('filt_size should be odd')
        return

    N2 = (N-1)/2
    x = np.linspace(-N2,N2,N)
    h = norm.pdf(x/sigma)

    return np.outer(h,h)

def keep_valid_image(x,keep_shape):
    """ Keeps valid portion of filtered image.

        inputs:
            x          -> filterd image
            keep_shape -> (N1,N2) dimensions of valid image
                          (should be same as original image)
        
        outputs:
            y -> N1-by-N2 valid image
    """
    return x[-keep_shape[0]-1:-1,-keep_shape[1]-1:-1]
def keep_valid_images(x,keep_shape):
    """ Keep valid portion of filtered images """

    images = []
    for i in tqdm(x):
        images.append(keep_valid_image(i,keep_shape))
    return np.array(images)
