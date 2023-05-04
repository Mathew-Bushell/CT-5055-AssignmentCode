import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import sklearn
from sklearn.preprocessing import MinMaxScaler

def verifymissing():
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

def modelTrain():
    # unpacks and formats the dataset, sorts it into date and items brought
    buyHistory = pd.read_csv("buying_history.csv")
    date = []
    trainingCount=[]
    validationCount=[]
    info = buyHistory['Year;Month;ItemsBought'].tolist()
    for i in info:
        row = i.split(";")
        date.append(row[0]+"-"+row[1])
        trainingCount.append(row[2])
        validationCount.append(row[2])
    # date = pd.to_datetime(date, format='%Y-%m')
    trainData = {'Brought': trainingCount}
    trainingFrame = pd.DataFrame(data=trainingCount)
    # print(trainingFrame)
    tf.random.set_seed(7)
    trainingSet = trainingFrame.values
    trainingSet = trainingSet.astype('float32')
    # normalises the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    trainingSet = scaler.fit_transform(trainingSet)

    # spits the training set into training and validation
    trainSize = int(len(trainingSet) * 0.7)
    validSize = len(trainingSet) - trainSize
    train, validation = trainingSet[0:trainSize, :], trainingSet[trainSize:len(trainingSet), :]
    print("Training = "+str(len(train)), "Validation = " + str(len(validation)))



modelTrain()
# verifymissing()