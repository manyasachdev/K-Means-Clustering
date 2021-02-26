import math

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __str__(self):
        return (f'({self.x}, {self.y})')

#Example points
p1=Point(20,5)
pc=Point(6,-1)

p2 =Point(4,2)
p3 =Point(7,8)

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
    

print(E_Distance(p1, pc))
print(newCentroid)
print(p4)






