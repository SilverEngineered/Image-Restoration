from pyspark import SparkContext
import numpy as np
import argparse
import crossValidation as cv
import imageProcessing as ip
import preProcess as pp
import time
import csv
import os

def saveImagesFromRDD(z,x,y,num_imgs,np):
    """ Saves Images from an RDD of scaled down images

        inputs:
            z -> (index,restored image) where image is scaled [0,1]
            x -> (index,original image) where image is scaled [0,1]
            y -> (index,distorted image) where image is scaled [0,1]
            num_imgs -> the number of images to be saved
            np -> noise power
    """
    z_imgs = z.sortBy(lambda x: x[0]).take(num_imgs)
    x_imgs = x.sortBy(lambda x: x[0]).take(num_imgs)
    y_imgs = y.sortBy(lambda x: x[0]).take(num_imgs)
    for i in range(num_imgs):
        scaled_z = (z_imgs[i][1]*255).astype(int)
        scaled_x = (x_imgs[i][1]*255).astype(int)
        scaled_y = (y_imgs[i][1]*255).astype(int)
        pp.save_image(scaled_z,"./output_images/restored/" + str(i) + "_np=" + str(np)+".jpg")
        pp.save_image(scaled_x,"./output_images/original/" + str(i) + "_np=" + str(np)+".jpg")
        pp.save_image(scaled_y,"./output_images/deg/" + str(i) + "_np=" + str(np)+".jpg")
def npToRDD(path,sc,N,num_images):
    """ Loads an np binary file and converts it to an RDD

        inputs:
            path: The file location and name of the .npy binary file

        outputs:
            RDD: (index,(np array)) where index is the original index of the image        

    """
    imgs = np.load(path)
    img_with_index = []
    for i in range(num_images):
        img_with_index.append((i,imgs[i]))
    return sc.parallelize(img_with_index).partitionBy(numPartitions=N)
def writeOutput(y_path,N,best_lam,time_elps,rss_list,lam_list,filename):
    """  Outputs analysis information to specified filename as csv file """
    with open(filename,'w') as csv_file:
        writer = csv.writer(csv_file,delimiter=',',quotechar='"')
        writer.writerow(["YRDD Input File", "Num Partitions", "Best Lambda", "Time Taken","Lambda List", "RSS"])
        data_row = [y_path,str(N),str(best_lam),str(time_elps),str(lam_list),str(rss_list)]
        writer.writerow(data_row)
        writer.writerow([])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_data_path", default='../data/mnist/mnist.npy')    
    parser.add_argument("--y_data_path", default='../data/mnist/deg/0.0.npy')
    parser.add_argument("--lambdas", default=[2* i for i in range(10)])    
    parser.add_argument("--k", type = int,default=10)
    parser.add_argument("--num_partitions", type = int,default=10)
    parser.add_argument("--save_image", default=True)
    parser.add_argument("--num_images", type=int, default=300)
    parser.add_argument("--num_imgs_to_save", default=5)
    args = parser.parse_args()

    noise_power_str = args.y_data_path.split('/')[-1]
    noise_power_str = noise_power_str[0:3]
    data_out = 'output_analysis/analysis_N_'+str(args.num_partitions)+'_noise_'+noise_power_str+'.csv'

    sc = SparkContext()

    xrdd = npToRDD(args.x_data_path,sc,args.num_partitions,args.num_images).cache()
    yrdd = npToRDD(args.y_data_path,sc,args.num_partitions,args.num_images).cache()
    joined = xrdd.join(yrdd).partitionBy(numPartitions=args.num_partitions).cache()
    start = time.time()
    rss_vals = cv.cross_validate(xrdd,yrdd,args.k,args.lambdas)
    best_lam = args.lambdas[np.argmin(rss_vals)]
    
    H = ip.estimate_filter_full(joined) 
    Z = yrdd.mapValues(lambda y: ip.restore_image(y,H,best_lam))

    time_elps = time.time() - start
    
    writeOutput(args.y_data_path,args.num_partitions,best_lam,time_elps, 
                args.lambdas,rss_vals,data_out)
    if args.save_image:
        if not os.path.exists("output_images/"):
            os.mkdir("output_images/")
        if not os.path.exists("output_images/restored"):
            os.mkdir("output_images/restored/")
        if not os.path.exists("output_images/original"):
            os.mkdir("output_images/original/")
        if not os.path.exists("output_images/deg"):
            os.mkdir("output_images/deg/")
        saveImagesFromRDD(Z,xrdd,yrdd,args.num_imgs_to_save,noise_power_str)
