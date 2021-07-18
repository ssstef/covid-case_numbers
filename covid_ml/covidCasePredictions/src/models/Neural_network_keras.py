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
import tensorflow as tf
import pandas as pd

# Delete the ones I don't use later!
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
import tensorflow
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy


#load new dataset and start with actual analyses
dataset = read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join_cases.csv', header=0, index_col=0)

# load dataset this dataset only if you want to use binary infomation for Neural Net (increasing vs decreasnig case numbers)
#dataset = read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join1.csv', header=0, index_col=0)
################################################################################################################
# Do this to use new cases instead of R_kat as dependent variable, otherrwise comment out!
# #create 'new variable and delete the old one
# dataset['new_cases'] = dataset['new_cases_smoothed']
#
# dataset.drop(['new_cases_smoothed'], axis=1, inplace=True)
#
# # save dataset
# dataset.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join_cases.csv')

#load new dataset and start with actual analyses
dataset = read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join_cases.csv', header=0, index_col=0)
################################################################################################################
#values = dataset.values
values = dataset.to_numpy()
# integer encode direction
encoder = LabelEncoder()
#values[:, 4] = encoder.fit_transform(values[:, 4])  # currently not in use because I removed non-numerical values

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)

print('Before reframe', dataset.head())

# frame as supervised learning
# Use 5 lags for all features, no leads (don't use future but past to predict the future cases)
reframed = series_to_supervised(scaled, 5, 1)


# I dropped the columns I don't want to predict or use for prediction in main_data.py
# drop columns we don't want to predict
#reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
print('Reframed, 5, 1', reframed.head())

# split into train and test sets
values = reframed.values
#n_train_hours = 365 * 24
#working with daily data
n_train_days = 450
n_test_days = 526
train = values[:n_train_days, :]
test = values[n_train_days:n_test_days, :]
#Also create subset to predict the y value  !! delete actual values?!
prediction = values[n_test_days: , :]  # now empty...
# split into input and outputs
# added for final model  --> came up with better solution, see above
# values_pandas = pd.DataFrame(values)
# x = values_pandas.iloc[:, values_pandas.columns!='new_cases_smoothed']
# y = values_pandas.iloc[:, values_pandas.colums == 'new_cases_smoothed']
# train_X, train_y = train[:, values_pandas.columns!='new_cases_smoothed'], train[:, values_pandas.colums == 'new_cases_smoothed']
# test_X, test_y = test[:, values_pandas.columns!='new_cases_smoothed'], test[:, values_pandas.colums == 'new_cases_smoothed']
#prediction_X, prediction_y = prediction[:, :-1], prediction[:, -1]

# Was R_kat
x = values[:, :-1]
y = values[:, -1]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
prediction_X, prediction_y = prediction[:, :-1], prediction[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
X_reshaped = x.reshape((x.shape[0], 1, x.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# # Different way of building model
#
# inputs = tf.random.normal([32, 10, 8])
# lstm = tf.keras.layers.LSTM(4)
# output = lstm(inputs)
# print(output.shape)

# lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
# whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
# print(whole_seq_output.shape)
#
# print(final_memory_state.shape)
# print(final_carry_state.shape)


# design network
no_classes = 1 # put output into 10 classes (does this make sense?!)

model = Sequential()
model.add(LSTM(60, input_shape=(train_X.shape[1], train_X.shape[2])))  # added activation = relu
###### model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), activation='relu'))  # added activation = relu
#####model.add(LSTM(128))
######model.add(MaxPooling2D(pool_size=(4, 50)))  # added this line (does this pool_size make any sense?!
model.add(Dense(1))
#######model.add(Dense(no_classes, activation='softmax'))  ## added this(end with softmax...) destroys the entire model..
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'] )
#model.compile(loss='mae', optimizer='adam')  ##this woks
metrics=['accuracy']
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)


####
#OR
###

# Create the model
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(no_classes, activation='softmax'))



# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2])) #test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# invert scaling for forecast
pred_scaler = MinMaxScaler(feature_range=(0, 1)).fit(dataset.values[:,0].reshape(-1, 1))
# Error: Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
yhat = yhat.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)
inv_yhat = pred_scaler.inverse_transform(yhat)
# invert scaling for actual
inv_y = pred_scaler.inverse_transform(test_y)

# Plot yhat against y
# plot history
pyplot.plot(inv_y, label='yhat Test')
pyplot.plot(inv_yhat, label='y Test')
pyplot.legend()
pyplot.show()

# Delte this later!
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# calculate RMSE
print('inv_y', inv_y)
print('inv_yhat', inv_yhat)
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# Generate generalization metrics --> Wie kann ich das anpassen, damit es die gewünschten Ergebnisse zeigt?
#score = model.evaluate(test_X, test_y, verbose=0)
#print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# Delete training model
del model

# Build final model with entire dataset
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_reshaped, y, epochs=250, batch_size=72, verbose=2,
                    shuffle=False)



############################################
# Save model (to use for predictions later)
from keras.models import load_model

#model.save('my_model2.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model



# make a prediction
yhat = model.predict(X_reshaped)
X = X_reshaped.reshape((X_reshaped.shape[0], X_reshaped.shape[2])) #test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# invert scaling for forecast
pred_scaler = MinMaxScaler(feature_range=(0, 1)).fit(dataset.values[:,0].reshape(-1, 1))
# Error: Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
yhat = yhat.reshape(-1, 1)
y = y.reshape(-1, 1)
inv_yhat = pred_scaler.inverse_transform(yhat)
# invert scaling for actual
inv_y = pred_scaler.inverse_transform(y)

# Plot yhat against y
# plot history
pyplot.plot(inv_y, label='yhat')
pyplot.plot(inv_yhat, label='y')
pyplot.legend()
pyplot.show()


# Plot yhat against y
# plot history
# define figure size
pyplot.figure(figsize=(8, 8))
pyplot.plot(yhat, label='yhat')
pyplot.plot(y, label='y')
pyplot.legend()
pyplot.show()
pyplot.savefig('y and yhat entire dataset')

# Now predict data for the last 30 days
# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')
#model.predict()
