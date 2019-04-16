from pyspark import SparkContext as sc
import numpy as np
import argparse

def npToRDD(path):
    """ Loads an np binary file and converts it to an RDD

        inputs:
            path: The file location and name of the .npy binary file

        outputs:
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='../data/mnist/')    
    args = parser.parse_args()

