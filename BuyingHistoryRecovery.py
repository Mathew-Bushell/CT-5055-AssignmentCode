import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
    # reshape into X=t and Y=t+1
    lookBack = 1
    trainingX, trainingY = createDataset(train, lookBack)
    validationX, validationY = createDataset(validation, lookBack)
    #reshapes the inputs to be [samples, time steps, features]
    trainingX = np.reshape(trainingX, (trainingX.shape[0], 1, trainingX.shape[1]))
    validationX = np.reshape(validationX, (validationX.shape[0], 1, validationX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, lookBack)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainingX, trainingY, epochs=100, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(trainingX)
    testPredict = model.predict(validationX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainingY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([validationY])
    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(trainingSet)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookBack:len(trainPredict) + lookBack, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(trainingSet)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (lookBack * 2) + 1:len(trainingSet) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(trainingSet))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

def createDataset(dataset, lookBack=1):
    dataX = []
    dataY = []
    for i in range(len(dataset)-lookBack-1):
        a = dataset [i:(i+lookBack), 0]
        dataX.append(a)
        dataY.append(dataset[i + lookBack, 0])
    return np.array(dataX), np.array(dataY)

modelTrain()
# verifymissing()