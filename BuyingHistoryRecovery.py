file = open("buying_history.csv", "r")
file = file.readlines()
lastYear = "2009"
lastMonth = "12"

for line in file:
    if line == "Year; Month;ItemsBought\n":
        continue
    else:
        line = line.replace("\n", "")
        line = line.split(";")
        if lastMonth == "12":
            lastMonth = "0"
        print(lastYear+";"+lastMonth+">"+line[0]+";"+line[1])
        if int(line[1]) == (int(lastMonth) + 1):
            print("Month good")

            if int(line[0]) == (int(lastYear) + 1) and lastMonth == "0":
                print("NextYear good")
                lastMonth = line[1]
                lastYear = line[0]
            elif (line[0] == lastYear):
                print("Same Year good")
                lastMonth = line[1]
            else:
                print("Year Bad")
        else:
            print("Month Bad")