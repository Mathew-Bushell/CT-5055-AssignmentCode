# class Coords:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
import math
replace = ("")
links = open("Cities.txt", "r")
linksCont = links.read()
splitLinks = linksCont.split("\n")
locations = open("location.txt", "r")
locationsCont = locations.read()
splitLocations = locationsCont.split("\n")


for line in splitLinks:
    lineSplit = line.split(",")
    for city1 in splitLocations:
        # print(city1, "pt1")
        city1Split = city1.split(";")
        # print(citySplit[0], lineSplit[0])
        if city1Split[0] == lineSplit[0]:
            L1x = city1Split[1]
            L1y = city1Split[2]
            break
    for city2 in splitLocations:
        city2Split = city2.split(";")
        # print(city2, "pt2")
        # print(citySplit[0], lineSplit[1])
        if city2Split[0] == lineSplit[1]:
            L2x = city2Split[1]
            L2y = city2Split[2]
            break


    Dx = (float(L2x) - float(L1x))
    Dy = (float(L2y) - float(L1y))
    distance = math.sqrt((Dx * Dx)+(Dy * Dy))
    lineadd = (line+","+str(distance)+"\n")
    replace = replace+lineadd
print (replace)
links.close()
locations.close()
newLinks = open("Cities.txt", "w")
newLinks.write(replace)
newLinks.close()