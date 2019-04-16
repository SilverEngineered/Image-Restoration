import preProcess as pp
import imageProcessing as ip
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--picklepath', default='../data/mnist/mnist.pkl')
    parser.add_argument('--numpypath', default='../data/mnist/')
    parser.add_argument('--scaling', default=[0,1])
    parser.add_argument('--noise_powers', default=[.1* i for i in range(10)])
    parser.add_argument('--img_size', default=[28,28])
    parser.add_argument('--filter_size',default=15)
    parser.add_argument('--sigma', default=3)
    args = parser.parse_args()

    pad_shape=(args.filter_size-1)/2
    mnist = pp.import_images(args.picklepath)
    scaled = pp.scale_images(mnist,scaling=args.scaling)
    pp.save_all_images(scaled,args.numpypath + "/mnist") #Scaled mnist images

    print("Mnist Images Saved")
    padded = pp.pad_images(scaled,pad_shape)
    print("Padding Complete")
    if not os.path.exists(args.numpypath + "deg/"):
        os.mkdir(args.numpypath + "deg/")
    for noise in args.noise_powers:
        degraded = ip.degrade_images(padded,args.sigma,args.filter_size,noise)
        degraded = ip.keep_valid_images(degraded,args.img_size)
        
        pp.save_all_images(degraded,args.numpypath + "deg/"+  str(noise))
        print("Degraded Image " + str(noise) + " saved")
    
    


    
