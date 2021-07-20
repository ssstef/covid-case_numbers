import sys
sys.path.append("src/models/")
sys.path.append("src/data/")
sys.path.append("data/processed")

import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
dataset = pd.read_csv('/covid_ml/covidCasePredictions/data/processed/join1.csv') #, usecols=[1], engine='python')
plt.plot(dataset)
#plt.show()


# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
dataframe = pd.read_csv('/covid_ml/covidCasePredictions/data/processed/join1.csv')  #, usecols=[1], engine='python')

dataset = dataframe.values

dataset = dataset.astype('float32')

data =pd.DataFrame(dataset)
print(data.head())

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
dataset2 = scaler2.fit_transform(np.reshape(dataset[:,0],(dataset.shape[0],1)))

# split into train and test sets: orde is important --> no random split
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print('Len train and test')
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0:]  #10 weg --> verwerndet alle Spalten
		#print('a', a)
		dataX.append(a)
		#print('dataX', dataX)
		dataY.append(dataset[i + look_back, 0])
		#print('dataY', dataY)
	return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 2
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


print( 'VOR', trainX.shape[0],trainX.shape[1],trainX.shape[2])
print( 'VOR', trainY.shape[0])

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]*trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]*testX.shape[2]))

print( 'Nach', trainX.shape[0],trainX.shape[1],trainX.shape[2])
print( 'Nach', trainY.shape[0])
#print( 'Dataset1', dataset.shape[0],dataset.shape[1])
#print( 'Dataset2', dataset2.shape[0],dataset2.shape[1])


# create and fit the LSTM network
model = Sequential()
#model.add(LSTM(4, input_shape=(1, look_back*10)))
model.add(LSTM(1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)  ##!! Anzahl der Epochen später wieder höher setzen

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler2.inverse_transform(trainPredict) #scaler oder scaler2?
trainY = scaler2.inverse_transform([trainY])
testPredict = scaler2.inverse_transform(testPredict)
testY = scaler2.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))  #Train Score: 1531.86 RMSE
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore)) # Test Score: 1127953.07 RMSE

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
print('Trainpredict', trainPredictPlot.shape())

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))  # scaler2 macht alles noch schlimmer
plt.plot(dataset)
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.legend()
#plt.legend((testPredictPlot, trainPredictPlot), ('testPredictPlot 1', 'trainPredictPlot'))
plt.show()

# cite: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/