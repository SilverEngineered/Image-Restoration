from pyspark import SparkContext
import numpy as np
import argparse
import crossValidation as cv
import imageProcessing as ip
import preProcess as pp
import time
import csv
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
def writeOutput(y_path,N,lam,time_elps,filename):
    """  Outputs analysis information to specified filename as csv file """
    with open(filename) as csv_file:
        writer = csv.writer(csv_file,delimiter=',',quotechar='"')
        writer.writerow("YRDD Input File", "Num Partitions", "Best Lambda", "Time Taken")
        writer.writerow([y_path,N,lam,time_elps])
        writer.writerow()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_data_path", default='../data/mnist/mnist.npy')    
    parser.add_argument("--y_data_path", default='../data/mnist/deg/0.0.npy')
    parser.add_argument("--lambdas", default=[.1* i for i in range(10)])    
    parser.add_argument("--k", type = int,default=10)
    parser.add_argument("--num_partitions", default=10)
    parser.add_argument("--data_out", default="analysis.csv")
    parser.add_argument("--save_image", default=False)
    parser.add_argument("--num_images", default=300)
    parser.add_argument("--img_index_to_save", default=0)
    args = parser.parse_args()

    sc = SparkContext()

    xrdd = npToRDD(args.x_data_path,sc,args.num_partitions,args.num_images)
    yrdd = npToRDD(args.y_data_path,sc,args.num_partitions,args.num_images)
    joined = xrdd.join(yrdd)
    start = time.time()
    rss_vals = cv.cross_validate(xrdd,yrdd,args.k,args.lambdas)
    best_lam = lambdas[np.argmin(rss_vals)]

    H = ip.estimate_filter_full(joined)
    Z = ip.restore_image(yrdd,best_lam,H)

    time_elps = start - time.time()
    writeOutput(args.y_data_path,args.num_partitions,best_lam,time_elps,args.data_out)
    if args.save_image:
        img = pp.scale_images(Z.collect(),scaling=[0,255])[0]
        pp.save_image(img)
