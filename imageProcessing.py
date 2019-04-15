import numpy as np

def deg_function(x,filt_param,filt_size,noise_power)
    """ Performs artificial degradation to a single image.
        The image is first blurred then Gaussian noise is added.

        inputs:
            x           -> input image np array (zero padded)
            filt_param  -> parameter determining the size of the blur filter
            filt_size   -> size of square filter support
            noise_power -> variance of additive Gaussian noise
        outputs:
            y -> degraded output image
    """
    pass
def filter(x,filt_param,filt_size,filt_type="gauss")
    """ Applies linear filter.

        inputs:
            x          -> input image np array (zero padded)
            filt_param -> parameter determining the size of the blur filter
            filt_size  -> size of square filter support
            filt_type  -> filter shape (default=gauss)
        outputs:
            y -> filtered image
    """
    pass
def inverse_filter(x,filt_param,filt_size,filt_type="gauss",lam=0)
    """ Applies linear inverse filter with regularization.

        inputs:
            x          -> input image np array (zero padded)
            filt_param -> parameter determining the size of the blur filter
            filt_size  -> size of square filter support
            filt_type  -> filter shape (default=gauss)
            lam        -> regularization parameter
        outputs:
            y -> filtered image
    """
    pass
def add_noise(x,noise_power)
    """ Adds white Gaussian noise to image.

        inputs:
            x           -> input image np array
            noise_power -> variance of additive noise
        outputs:
            y -> noisy image
    """
    size_x = x.shape()
    noise = np.random.normal(0,np.sqrt(noise_power),size_x)
    return np.add(x,noise)
def gauss_filter(filt_param,filt_size)
    """ Returns Gaussian filter impulse response as np array.

        inputs:
            filt_param -> parameter determining the size of the blur filter
            filt_size  -> size of square filter support
        outputs:
            h -> Gaussian filter impulse response (filt_size-by-filt_size)
    """
    pass
