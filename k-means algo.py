import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

#Example points
p1=Point(20,5)
pc=Point(6,-1)

#Calculating Euclidean Distance
def E_Distance(P1, Pc):
    sum = math.sqrt(math.pow((P1.x-Pc.x),2) + math.pow((P1.y-Pc.y),2))
    return sum

print(E_Distance(p1, pc))




