import numpy as np
import preProcess as pp
import imageProcessing as ip

img = pp.import_images()[0]

im_size = (512,512)
im0 = np.zeros(im_size)
print im0
noise_power = 1
noise = ip.add_noise(im0,noise_power)
print noise

noise_scale = pp.scale_images(noise,scaling=[0,255],dtype=np.uint8)
pp.save_image(noise_scale,"test_images/noise_512.jpg")

sigma = 5
Nfilt = 5*sigma
print 'filter size: ',Nfilt
H = ip.gauss_filter(sigma,Nfilt)
H_scale = pp.scale_images(H,scaling=[0,255],dtype=np.uint8)
pp.save_image(H_scale,"test_images/H_%.2f.jpg" % sigma)

Nfilt2 = (Nfilt-1)/2
print 'Nfilt2: ',Nfilt2

noise_pad = np.pad(noise,(Nfilt2,Nfilt2),'constant')
print 'upadded size: ',noise.shape
print 'padded size: ',noise_pad.shape

y = ip.filter_image(noise_pad,sigma,Nfilt)
y_scale = pp.scale_images(y,scaling=[0,255],dtype=np.uint8)
pp.save_image(y_scale,'test_images/y.jpg')

lam = 1
yi = ip.inverse_filter_image(y,sigma,Nfilt,lam=lam)
yi_scale = pp.scale_images(yi,scaling=[0,255],dtype=np.uint8)
pp.save_image(yi_scale,'test_images/yi_%.2f.jpg' % lam)

deg_img = ip.degrade_image(noise_pad,sigma,Nfilt,noise_power)
deg_scale = pp.scale_images(deg_img,scaling=[0,255],dtype=np.uint8)
pp.save_image(deg_scale,'test_images/deg_img_%.2f.jpg' % noise_power)
