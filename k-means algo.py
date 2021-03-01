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
    def __init__(self, x = 0, y = 0, color = 'hotpink'):
        self.x = int(x)
        self.y = int(y)
        self.color = color

    def __str__(self):
        return f"({self.x}, {self.y})"


# Cluster Class
class Cluster:
    def __init__(self, centroid = Point(0, 0), sse = 0, color = "white"):
        self.centroid = centroid
        self.points = []
        self.sse = sse
        self.dists = []
        self.color = color

    def __str__(self):
        pointList = ""
        for i in range(len(self.points)):
            pointList = pointList + f"({self.points[i].x}, {self.points[i].y})  "
        return pointList

    # Add point to the cluster
    def addPoint(self, newPoint):
        self.points = self.points + [newPoint]

    # Add distance 
    def addDistance(self, distance):
        self.dists = self.dists + [distance]

    # Change the centroid of the cluster
    def changeCentroid(self, newPoint):
        self.centroid.x = (newPoint.x + self.centroid.x) / 2
        self.centroid.y = (newPoint.y + self.centroid.y) / 2

    # Plot the cluster using Pyplot
    def plotCluster(self):
        x_list = []
        y_list = []        
        for i in range(len(self.points)):
            x_list.append(self.points[i].x)
            y_list.append(self.points[i].y)
        
        plt.scatter(x_list, y_list, color = self.color)
    
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
    distance = math.sqrt(
        math.pow((dataPoint.x - centroid.x), 2)
        + math.pow((dataPoint.y - centroid.y), 2)
    )
    return distance


# Compares EuclideanDistances and returns the position of the chosen cluster
def compareEuclideanDistance(distances):
    #position = 0
    for i in range(len(distances)):
        if distances[i] == min(distances):
            return i

# Test function to print and check different values
def dryTest(clusterSet, DataSet):

    # Printing the dataset
    print("DataSet:")
    for point in DataSet:
        print(point)

    # Displaying the clusters
    for cluster in clusterSet:
        print("Cluster------")
        print(f"Centroid = ({cluster.centroid.x}, {cluster.centroid.y})")
        print(cluster)
        
# ------------------------------------------------------------------------------------------------------------------------------------

def main():
    # DEALING WITH DATASET----------------------------------------------------------------------------------------------------------------
    # Parsing the CSV file using Numpy
    x, y = np.loadtxt("testcase.csv", unpack=True, delimiter=",")
    print(x)
    print(y)
    plt.scatter(x, y, color = 'hotpink')
    plt.show()

    # Adding data to a tuple of Points
    DataList = []
    for i in range(len(x)):
        DataList.append(Point(x[i], y[i]))

    DataTuple = tuple(DataList)

    print("DataTuple:")
    for point in DataTuple:
        print(point) 
    # ------------------------------------------------------------------------------------------------------------------------------------

    # IMPLEMENTATION---------------------------------------------------------------------------------------------------------------------
    # K-Means Clustering Algorithm
    
    # Initialize K
    sumSqDistList = [] # List of SSEs
    K_list =[]
    idealKClusters = [] # Clusters of the ideal K

    # Loop from 1 to 10, for each value of K
    for m in range(1,4):
        K = m

        # clusters = List of Cluster objects
        clusters = []
        colors = ["black", "red", "pink", "yellow", "blue", "magenta", "orange", "brown", "green", "purple"]

        # Select random centroids and create a list of K clusters
        randomPoints = selectRandomPoints(K, DataTuple)
        for point in randomPoints:
            print("RandomPoints")
            print(point)

        # Creating and initializing the clusters using random centroids
        for i in range(K):
            clusterInstance = Cluster(centroid=randomPoints[i], color=colors[i])
            clusters.append(clusterInstance)

        # Iterate throught the dataset
        for point in DataTuple:
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

            # Update the centroid of the appended cluster
            clusters[clusterPosition].changeCentroid(point)

            ######################################`
        idealKClusters.append(clusters)
        #All points created and assigned
        
    #Calculating SSEs for all clusters
    temp2 = 0
    i = 0
    for cluster in clusters:
        for point in cluster.points:
            temp = calculateEuclideanDistance(point, cluster.centroid)
            temp = math.pow(temp, 2)
            cluster.sse = temp
        temp2 = temp2 + cluster.sse
        sumSqDistList.append(temp2)
        K_list.append(i+1)
        i = i + 1

    # Elbow Method
    # l1 and l2 are the start and end points of the line joining the first and the last pints of the SSE scores
    l1 = np.array([K_list[0],sumSqDistList[0]])
    l2= np.array([K_list[len(K_list)-1],sumSqDistList[len(sumSqDistList)-1]])
    # Plot (k, sse) and find the ED for each point from the line.
    max = 0
    idealK = 1
    for z in range(K):
        # Convert each point to numpy arrays
        p3 = np.array([K_list[z],sumSqDistList[z]])
        # Distance between line and point
        d = abs(np.cross(l2-l1,p3-l1)/np.linalg.norm(l2-l1))
        # k with the maximum ED is the ideal K
        if (d > max):
            max = d
            idealK = K_list[z]
    
    print("DataTuple:")
    for point in DataTuple:
        print(point) 

    print(f"IDEAL K: {idealK}")

    

    # Plot the clusters for the ideal K
    for cluster in idealKClusters[idealK]:
        cluster.plotCluster()

    plt.show()
    # ---------------------------------------------------------------------------------------------------------------------------------

# Calling main
main()

#---------------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------------------