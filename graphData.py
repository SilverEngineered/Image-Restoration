from matplotlib import pyplot as plt
import os
import csv


def getContentFromFile(l1,datapath,field):
    points = []
    with open(datapath + l1) as csv_file:
        reader = csv.reader(csv_file,delimiter=',')
        count = 0
        for row in reader:
            if count==1:
                points=row
            count+=1
    return points[field]
datapath="./output_analysis/"
files = os.listdir(datapath)
n5=list(filter(lambda x: ".5" in x,files))
n1=list(filter(lambda x: ".1" in x,files))
datapath2="./output_analysis2/"
lamrssFiles = os.listdir(datapath2)
lamrssFiles.sort()

#Fields:
#0 = y Data path
#1 = Num Partitions
#2 = k
#3 = NumImages
#4 = Best Lambda
#5 = Time
#6 = Lambda List
#7 = RSS

numParts = []
times = []
n1.sort()
n5.sort()
for i in n1:
    x=int(getContentFromFile(i,datapath,1))
    numParts.append(x)
    x=float(getContentFromFile(i,datapath,5))
    times.append(x)
#print(numParts)
#print(times)

numParts5 = []
times5 = []
for i in n5:
    x=int(getContentFromFile(i,datapath,1))
    numParts5.append(x)
    x=float(getContentFromFile(i,datapath,5))
    times5.append(x)

lambdas=[2*i for i in range(20)]
rss_vals = []
for i in lamrssFiles:
    x=getContentFromFile(i,datapath2,7)[1:-1]
    x=x.split(',')
    x = [float(i) for i in x]
    rss_vals.append(x)

plot_val="partitions"
plot_val="rss"
if(plot_val=="partitions"):
    plt.plot(numParts,times,label='noise_power=.1')
    plt.plot(numParts5,times5,label='noise_power=.5')
    plt.title("Number of Partitions vs Time")
    plt.xlabel("Number of Partitions")
    plt.ylabel("Time to complete 1000 images (s)")
    plt.legend()
    plt.show()
    
if(plot_val=="rss"):
    for i in range(len(rss_vals)):
        label="noise_power" + str(.1*i)
        plt.plot(lambdas,rss_vals[i],label=label)
    plt.title("Lambda vs RSS Values")
    plt.xlabel("Lambda Value")
    plt.ylabel("RSS Value")
    plt.legend()
    plt.show()





