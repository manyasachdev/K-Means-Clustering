# CS-2376
# Lab Assignment-1: Clustering
# Team Members: Manya Sachdev, Pratham Singh, Shivani

import math
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

# CLASSES---------------------------------------------------------------------------------------------------------------
# Point Class
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"


# Cluster Class
class Cluster:
    def __init__(self, centroid=Point(0, 0), color="white"):
        self.centroid = centroid
        self.points = []
        self.color = color

    def __str__(self):
        pointList = ""
        for i in range(len(self.points)):
            pointList = pointList + f"({self.points[i].x}, {self.points[i].y})  "
        return pointList

    # Add point to the cluster
    def addPoint(self, newPoint):
        self.points = self.points + [newPoint]

    # Change the centroid of the cluster
    def changeCentroid(self, newPoint):
        self.centroid.x = (newPoint.x + self.centroid.x) / 2
        self.centroid.y = (newPoint.y + self.centroid.y) / 2

    # Plot the cluster using Pyplot
    def plotCluster(self):
        for i in range(len(self.points)):
            plt.scatter(self.points[i].x, self.points[i].y, color = self.color)
    
    # To plot the centroid of the cluster using Pyplot
    def plotCentroid(self):
        plt.scatter(self.centroid.x, self.centroid.y, color = 'yellow')


# ---------------------------------------------------------------------------------------------------------------------------------

# UTILITY FUNCTIONS---------------------------------------------------------------------------------------------------------------
# Selects and returns random K points in a list from a given dataset
def selectRandomPoints(K, dataSet):
    randomPoints = []
    for i in range(K):
        randomPoints.append(random.choice(dataSet))
    return randomPoints


# Calculating Euclidean Distance
def calculateEuclideanDistance(dataPoint, centroid):
    sum = math.sqrt(
        math.pow((dataPoint.x - centroid.x), 2)
        + math.pow((dataPoint.y - centroid.y), 2)
    )
    return sum


# Compares EuclideanDistances and returns the position of the chosen cluster
def compareEuclideanDistance(distances):
    position = 0
    for i in range(len(distances)):
        if distances[i] == min(distances):
            return i


# ------------------------------------------------------------------------------------------------------------------------------------

def main():
    # DEALING WITH DATASET----------------------------------------------------------------------------------------------------------------
    # Parsing the CSV file using Numpy
    x, y = np.loadtxt("testcase.csv", unpack=True, delimiter=",")
    print(x)
    print(y)

    # Adding data to a list of Points
    DataSet = []
    for i in range(10):
        DataSet.append(Point(x[i], y[i]))
    #     plt.scatter(DataSet[i].x, DataSet[i].y, color = 'hotpink')
    # plt.show()
    # ------------------------------------------------------------------------------------------------------------------------------------

    # IMPLEMENTATION---------------------------------------------------------------------------------------------------------------------
    # K-Means Clustering Algorithm
    # Initialize K
    K = 3

    # clusters = List of Cluster objects
    clusters = []
    colors = ["black", "red", "pink"]

    # Select random centroids and create a list of K clusters
    randomPoints = selectRandomPoints(3, DataSet)

    # Creating and initializing the clusters using random centroids
    for i in range(K):
        clusterInstance = Cluster(centroid=randomPoints[i], color=colors[i])
        clusters.append(clusterInstance)

    # Iterate throught the dataset
    for point in DataSet:
        ######################################
        # point is the currently selected Point

        # EuclideanDistances = list of Eculidean Distances calculated bw the point and different centroids
        EuclideanDistances = []
        # Calculate Euclidean distances of point and each of the cluster centroids
        # and store it in an array with the same positions as that of the clusters
        for cluster in clusters:
            ######################################
            ed = calculateEuclideanDistance(point, cluster.centroid)
            EuclideanDistances.append(ed)
            ######################################

        # Compare the Euclidean Distances and choose the shortest cluster
        clusterPosition = compareEuclideanDistance(EuclideanDistances)

        # Add point to the cluster with the shortest Euclidean Distance from point
        clusters[clusterPosition].addPoint(point)
        print(
            f"Appended Cluster {clusterPosition} with point {point}. Length = {len(clusters[clusterPosition].points)}. Current centroid = ({clusters[clusterPosition].centroid.x}, {clusters[clusterPosition].centroid.y})"
        )

        # Update the centroid of the appended cluster
        clusters[clusterPosition].changeCentroid(point)
        print(
            f"Updated centroid = ({clusters[clusterPosition].centroid.x}, {clusters[clusterPosition].centroid.y})"
        )
        ######################################
    # ---------------------------------------------------------------------------------------------------------------------------------

    # PLOTTING CLUSTERS----------------------------------------------------------------------------------------------------------------
    # Printing the clusters
    print("DatSet:")
    for point in DataSet:
        print(point)

    for cluster in clusters:
        print("Cluster------")
        print(cluster)

    # Displaying the clusters
    for cluster in clusters:
        print(f"Centroid = ({cluster.centroid.x}, {cluster.centroid.y})")
        cluster.plotCentroid()
        cluster.plotCluster()



    plt.show()
    plt.close()
    # ----------------------------------------------------------------------------------------------------------------------------------

# Calling main
main()