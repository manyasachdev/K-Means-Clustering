import math
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

#Point Class
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __str__(self):
        return (f'({self.x}, {self.y})')

#Cluster Class
class Cluster:
    def __init__(self, centroid = Point(0,0), data = [], color = 'white'):
        self.centroid = centroid
        self.data = data
        self.color = color
    
    def __str__(self):
        pointList = ''
        for i in range(len(self.data)):
            pointList = pointList + f'({self.data[i].x}, {self.data[i].y})  '
        return pointList


        return (f'{self.data}')

    #Change the centroid of the cluster
    def changeCentroid(self, newPoint):
        self.centroid.x = (newPoint.x + self.centroid.x)/2
        self.centroid.y = (newPoint.y + self.centroid.y)/2

    def plotCluster(self):
        for i in range(len(self.data)):
            plt.scatter(self.data[i].x, self.data[i].y, color = self.color)


#Selects and returns random k points in a list from a given dataset
def selectRandomPoints(K, dataSet):
    randomPoints = []
    for i in range(K):
        randomPoints.append(random.choice(dataSet))
    return randomPoints


#Calculating Euclidean Distance
def calculateEuclideanDistance(dataPoint, centroid):
    sum = math.sqrt(math.pow((dataPoint.x-centroid.x),2) + math.pow((dataPoint.y-centroid.y),2))
    return sum

#Compares EuclideanDistances and returns the position of the chosen cluster 
def compareEuclideanDistance(distances):
    position = 0
    for i in range(len(distances)):
        if distances[i] == min(distances):
            return i


#Parsing the CSV file using Numpy
x,y = np.loadtxt('testcase.csv', unpack=True, delimiter=',')

#Adding data to a list of Points
DataSet=[]
for i in range(10):
    DataSet.append(Point(x[i], y[i]))
    plt.scatter(DataSet[i].x, DataSet[i].y, color = 'hotpink')
# plt.show()

#K-Means Clustering Algorithm

#Initialize K
K = 2

# cluster = List of Cluster objects
clusters = []
colors = ['black', 'hotpink', 'red']

#Select random centroids and create a list of K clusters
randomPoints = selectRandomPoints(2, DataSet)
for i in range(K):
    clusters.append(Cluster(centroid = randomPoints[i], color = colors[i]))

# print(f'{clusters[0].centroid}, {clusters[1].centroid}')

#Iterate throught the dataset
for i in range(len(DataSet)):

    # p = current point
    p = DataSet[i]

    # edList = list of Eculidean Distances calculated bw the point and different centroids
    EuclideanDistances = []
    #Calculate Euclidean distance of p, c1 and c2
    for k in range(K):
        ed = calculateEuclideanDistance(p, clusters[k].centroid)
        EuclideanDistances.append(ed)
    
    # Compare the Euclidean Distances and choose the shortest one
    clusterPosition = compareEuclideanDistance(EuclideanDistances)

    # Add p to the cluster with the shortest ED
    clusters[clusterPosition].data.append(p)

    # Update the centroid of the appended cluster
    clusters[clusterPosition].changeCentroid(p)

# Priting the clusters
for i in range (K):
    print(f'Cluster {i}:')
    print(clusters[i])

# plt.show