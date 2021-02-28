import math
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style



class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __str__(self):
        return (f'({self.x}, {self.y})')


x,y = np.loadtxt('testcase.csv', unpack=True, delimiter=',')

P_arr=[]
for i in range(10):
    P_arr.append(Point(x[i], y[i]))

for obj in P_arr: 
    print( obj.x, obj.y, sep =',' )

plt.plot(x, y)
plt.title('Sample Chart')
plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')

plt.show()


#Example points
p1=Point(20,5)
pc=Point(6,-1)

p2 =Point(4,2)
p3 =Point(7,8)

#Select random k points from a given dataset
def selectRandomPoints(K, dataSet):
    randomPoints = []
    for i in range(K):
        randomPoints.append(random.choice(dataSet))


#Calculating Euclidean Distance
def E_Distance(P1, Pc):
    sum = math.sqrt(math.pow((P1.x-Pc.x),2) + math.pow((P1.y-Pc.y),2))
    return sum

#Change the centroid
def changeCentroid(p1, p2):
    centroid = Point()
    centroid.x = (p1.x + p2.x)/2
    centroid.y = (p1.y + p2.y)/2
    return centroid

newCentroid = changeCentroid(p1, pc)

#Compare EuclideanDistance
def compareEuclideanDistance(d1, d2):
    if (d1  <= d2):
        return d1
    else:
        return d2

D1= E_Distance(p1,pc)
D2= E_Distance(p2,p3)
p4 = compareEuclideanDistance(D1, D2)
    

#print(E_Distance(p1, pc))
#print(newCentroid)
#print(p4)






