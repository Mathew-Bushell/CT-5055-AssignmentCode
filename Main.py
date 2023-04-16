import math
cityfile = open("location.txt", "r")
cities = cityfile.readlines()
start = ""
cityHeuristic = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "H": 0, "I": 0, "J": 0, "K": 0, "L": 0, "M": 0, "N": 0, "O": 0, "P": 0, "Q": 0, "R": 0, "S": 0, "T": 0, "U": 0, "V": 0, "W": 0, "X": 0, "Y": 0, "Z": 0}
#create dictionary for heuristics
startFound = False
while startFound != True:
    start = input("input your start position:")
    start = start.upper()
    for x in cities:
        city= x.split(";")
        print(city[0])
        if start == city[0]:


            startFound =True
            break

targetFound = False
while targetFound != True:
    target = input("input your target position:")
    target = target.upper()
    for x in cities:
        city= x.split(";")
        print(city[0])
        if target == city[0]:
            L1x = city[1]
            L1y = city[2]
            L1y = L1y.replace("\n", "")

            targetFound =True
            break

for startCity in cities:
    startCity = startCity.replace("\n","")
    citySplit = startCity.split(";")


    L2x = citySplit[1]
    L2y = citySplit[2]

    Dx = (float(L2x) - float(L1x))
    Dy = (float(L2y) - float(L1y))
    distance = math.sqrt((Dx * Dx)+(Dy * Dy))
    cityHeuristic[citySplit[0]] = distance
    print("Distance to ", citySplit[0], "is ", cityHeuristic[citySplit[0]])