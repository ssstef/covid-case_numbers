import os
import sys
sys.path.append("src/models/")
sys.path.append("src/data/")

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import nn_functions
from nn_functions import series_to_supervised

# Prediction using LSTM Network
from keras.models import load_model


from pandas import read_csv  # warum bekommen ich eine Fehlermeldung bei Zeile 18 wenn ich das nicht importiere, aber es wird trotzdem ein Modell trainiert?!


# if time: use dataset for Austria to predict
# for now: predict the latest months of data

# load dataset
dataset = read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join1.csv', header=0, index_col=0)


#values = dataset.values
values = dataset.to_numpy()
# integer encode direction
encoder = LabelEncoder()
#values[:, 4] = encoder.fit_transform(values[:, 4])  # currently not in use because I removed non-numerical values

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))  # or should I import scaler??
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
# I dropped the columns I don't want to predict or use for prediction in main_data.py
#reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
#n_train_hours = 365 * 24
#working with daily data
n_train_days = 400
n_test_days = 450
train = values[:n_train_days, :]
test = values[n_train_days:n_test_days, :]
#Also create subset to predict the y value  !! delete actual values?!
prediction = values[n_test_days: , :]
# split into input and outputs
# added for final model
x = values[:, :-1]
y = values[:, -1]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
prediction_X, prediction_y = prediction[:, :-1], prediction[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# for final model!
X = x.reshape((x.shape[0], 1, x.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# Use saved model from Neural_network_keras.py
#model = load_model('my_model.h5')

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
pred_scaler = MinMaxScaler(feature_range=(0, 1)).fit(dataset.values[:,0].reshape(-1, 1))
# Error: Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
yhat = yhat.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)
inv_yhat = pred_scaler.inverse_transform(yhat)
# invert scaling for actual
inv_y = pred_scaler.inverse_transform(test_y)
