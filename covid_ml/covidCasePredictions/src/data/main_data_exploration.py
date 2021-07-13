# Import modules

import os
import sys
sys.path.append("src/models/")
sys.path.append("src/data/")
sys.path.append("src/visualization/")


from datetime import date, datetime

import pandas as pd
from covid_ml.covidCasePredictions.src.features.build_features import *
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
plt.style.use('ggplot')  # Make figures pretty
#To preprocess data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from visualize import correlation

print('In this file I do descriptive analyses on the covid data generated in main_data')

data = pd.DataFrame('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join1.csv')

correlation(data=data)


# # use read_excel to read part of larger dataset more quickly
# # Open and inspect data
# #All countries
# #df = pd.read_excel('covid_ml/covidCasePredictions/data/external/owid-covid-data.xlsx', sheet_name=0)
#
# #Only Germany (only date and numerical data)
# #df = pd.read_excel('covid_ml/covidCasePredictions/data/external/owid-covid-data-Germany-num.xlsx', sheet_name=0)
#
# #use subset of data
# df = read_excel('/covid_ml/covidCasePredictions/data/external/owid-covid-data.xlsx', nrows=150)
#
#
# #df = read_excel('covid_ml/covidCasePredictions/data/external/owid-covid-data.xlsx', nrows=150)
# df = pd.DataFrame(df)
#
#
# # Datensatz komplett anschauen
# pd.set_option("display.max_rows", None, "display.max_columns", None)
# print('see all colums')
# print(df.head())
#
# # Column names in dict and list
# col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(df.columns)]
# col_mapping_dict = {c[0]:c[1] for c in enumerate(df.columns)}
#
# print(col_mapping_dict)
#
#
# # use datetime to be able to work with dates
# df['date'] = pd.to_numeric(pd.to_datetime(df['date']))
# #df['date'] = df[str('date')]
#
# #df['date'] = pd.to_datetime(df['date'])  # , format='%Y%m%d'
#
# df.set_index(df['date'], inplace=True)
#
# #df = pd.DataFrame(df, index=date) #kann weg?
#
# # creating instance of one-hot-encoder
# # Zunächst reshape
#
#
#
# categorical = ['location', 'iso_code', 'continent']
# onehotlist= list()
#
# for i in categorical:
#     onehot_enc = OneHotEncoder()
#     i_spalte = df[i]
#     i_spalte_res = i_spalte.values.reshape(-1,1)
#     i_onehot = onehot_enc.fit_transform(i_spalte_res)
#     onehotlist.append([i, i_onehot, onehot_enc])
# print('Hier soll onehotlist stehen')
# print(onehotlist[0][2].categories_)
# #print(df['location'])
#
# enc_df = pd.DataFrame(onehotlist[0][1].toarray())
# #enc = OneHotEncoder(handle_unknown='ignore')
# # passing cat columns (label encoded values)
# #enc_df = pd.DataFrame(enc.fit_transform(df[['continent', 'location', 'iso_code']]).toarray())
# # merge with main df  on key values
# #df = df.join(enc_df)
# df_join = df.join(pd.DataFrame(enc_df.toarray(), columns=onehot_enc.categories_[0]))
#
# print('did join work?')
# df.set_index('date')
# print('did index setting work?')
# print(df.head())
#
#
# print('test')
# #for i in 'continent', 'location', 'iso_code':
# #    df[i] = pd.to_numeric(df[i],errors = 'coerce')
#
#
# print('is there categorical info left?')
# print(df.info())
#
# #Delete rows with missing data in relevant features
# df.dropna(how='any',
#           subset=['new_deaths', 'hosp_patients_per_million', 'new_vaccinations'])
#
#
#
# print(df.head())
# X = df.loc[:,df.columns != "new_deaths"]
# X = df.iloc[:, np.r_[0:3,15:19,24,25]]
# y = df["new_deaths"]
#
#
#
# # Missing Values in den Daten?
# print('null values in data')
# print(df.isnull().any(axis=1).sum())
#
# print('na values in data')
# print(df.isna().any(axis=1).sum())
#
# # print("Original Datafame: ")
# pd.set_option('display.max_rows', None, 'display.max_columns', 1000)
#
# print(df.head())
# print(df.shape)
# #Datatype info
# print('Datatype info')
# print(df.info(verbose=True))
#
# print(df.describe())
# ('Showing entire dataframe?')
# #print(df.describe(percentiles=[.1, .99]))
# #print(df.describe(include='all'))
#
# # Encoding of categorical data
#
# # Keep only rows without missing data
# # Drop rows which contain any NaN values --> empty data frame --> choose which infomation is necessary
# # mod_df = df.dropna()
#
# # Drop rows which contain any NaN value in the selected columns
# # mod_df = df.dropna(how='any',
# #                 subset=['new_deaths', 'hosp_patients_per_million', 'new_vaccinations'])
#
# # print("Modified Dataframe : ")
# # print(mod_df.head())
#
#
# # Data preprocessing
# #X = df.loc[:, df.columns != 'new_deaths']
# # try out with subse tof data
# X = df.loc[:, df.columns != "new_deaths"]
# y = df['new_deaths']
#
#
#
# from covid_ml.covidCasePredictions.src.data.Preprocessor import Preprocessor
# print('test new X')
# X_train, X_test, y_train, y_test = Preprocessor(X, y).get_data()
# print('using Preprocessor')
#
#
# # Descirptive analyses
# # Plot the data
#
# import plotly.graph_objs as go
# from plotly.offline import iplot
#
# plot = df.plot(style='.', figsize=(15, 8), title='Entire covid dataset')
# print(plot)
#
# import pandas as pd
#
# #df = df.set_index(['Datetime'])
# df.index = pd.to_datetime(df.index)
#
# if not df.index.is_monotonic:
#     df = df.sort_index()
#
# # df = df.rename(columns={'PJME_MW': 'value'})
# #plot_dataset(df, title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')
#
# # _ = df['new_deaths'].plot.hist(figsize=(15, 5), bins=200, title='Distribution of New Deaths')
#
# # plot_dataset(df, title='covid', xvalue="date", yvalue="new_deaths")
# # Models
#
# # Figures

# Remember for later projects
# To set up project structure using coockie cutter
from cookiecutter.main import cookiecutter
# doch nicht verwendetcookiecutter https://gitlab.com/ericdevost/cookiecutter-python.git
# cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science

# cookiecutter('cookiecutter-pypackage/')

# Plotly to create interactive graphs
