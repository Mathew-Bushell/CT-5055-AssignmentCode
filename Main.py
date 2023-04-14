cities = open("location.txt", "r")
startFound = False
#create dictionary for heuristics
while startFound == False:
    start = input("input your starting position:")
    start = start.upper()
    for x in cities:
        city= x.split(";")
        print(city[0])
        if start == city[0]:
            print("Yes")
            startFound =True
            break

targetFound = False
while targetFound == False:
    target = input("input your starting position:")
    target = target.upper()
    for x in cities:
        city= x.split(";")
        print(city[0])
        if start == city[0]:
            print("Yes")
            targetFound =True
            break


for x in cities:
    citySplit = startCity.split(";")
    for startCity in citySplit:
       # print(city1, "pt1")
       # city1Split = startCity.split(";")
       # print(citySplit[0], lineSplit[0])
       if startCity[0] == start:
            L1x = startCity[1]
            L1y = startCity[2]
            break
    for city2 in citySplit:
        # print(city2, "pt2")
        # print(citySplit[0], lineSplit[1])
        L2x = city2[1]
        L2y = city2[2]