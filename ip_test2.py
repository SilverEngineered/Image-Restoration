import numpy as np
import imageProcessing as ip
import preProcess as pp
import crossValidation as cv

x = pp.import_images()[0]
print 'x[0][0]: ',x[0][0]
y0 = np.load('../data/mnist/deg/0.0.npy')[0]
print 'y0[0][0]: ',y0[0][0]
y0_scale = pp.scale_images(y0,scaling=[0,255],dtype=np.uint8)
pp.save_image(y0_scale,'test_images/y0.jpg')

y3 = np.load('../data/mnist/deg/0.3.npy')[0]
y3_scale = pp.scale_images(y3,scaling=[0,255],dtype=np.uint8)
pp.save_image(y3_scale,'test_images/y3.jpg')

y9 = np.load('../data/mnist/deg/0.9.npy')[0]
y9_scale = pp.scale_images(y3,scaling=[0,255],dtype=np.uint8)
pp.save_image(y9_scale,'test_images/y9.jpg')

h = ip.estimate_filter(x,y0)
print 'h[0][0]: ',h[0][0]
h_scale = pp.scale_images(h,scaling=[0,255],dtype=np.uint8)
pp.save_image(h_scale,'test_images/h.jpg')

y0_hat = ip.restore_image(y0,h)
print 'y0_hat: ',y0_hat[0][0]

lamb = 0.9
print 'lambda: ',lamb
y9_hat = ip.restore_image(y9,h,lamb)
print 'max(y3_hat)',np.max(y9_hat)
y9_hat_scale = pp.scale_images(y9_hat,scaling=[0,255],dtype=np.uint8)
pp.save_image(y9_hat_scale,'test_images/y9_hat_%.4f.jpg' % lamb)

y0_hat_scale = pp.scale_images(y0_hat,scaling=[0,255],dtype=np.uint8)
pp.save_image(y0_hat_scale,'test_images/y0_hat.jpg')
