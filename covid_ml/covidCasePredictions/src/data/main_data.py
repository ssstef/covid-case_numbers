# Import modules
from datetime import date, datetime
import os
import sys
#sys.path.append("src/models/")
#sys.path.append("src/data/")
#sys.path.append("../data/")
#sys.path.append("data/..")
import pandas as pd
import plotly as plotly

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

import Preprocessor
from Preprocessor import leads

#'In this file I build datasets to analyse covid data'
# I build datasets for traditional machine learning - one to analyse a binary target variable: cases increasing or decreasing and one to predict case numbers
# Additionally I build a dataset to perform Deep Learning (LSTM)
# The main difference is the date structure and added lags/leads
# I use three data sources: weather, measures against Covid and a covid dataset including a vast number of statistics on cases, hospitalization, tests...some of these are highly correlated which is why I choose a subset of these

####################################################################################################
# First I load weather data and set the tiime variable as index to  merge on this later
# The weather data gets downloaded in get_weather_data.py in the daat folder.
#df_weather = pd.read_excel('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/external/weather_data3.xlsx')
# In case this doesn't work just use line above
df_weather = pd.read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/interim/weather_data2.csv')
print(df_weather.shape)

# Add Column names -- is this sttill necessary?
# df_weather.rename(columns=df_weather.iloc[0])
# new_header = df_weather.iloc[0]      #grab the first row for the header
# df_weather = df_weather[1:]          #take the data less the header row
# df_weather.columns = new_header      #set the header row as the df header

df_weather.set_index(df_weather['time'], inplace=True)  #!! eventuell wieder rein
print('is time now index?')


#df_weather_format = df_weather['time'].dt.
#df_weather['Date'] = floor_date.df_weather['time']
# Change date forrmat to match other datasets
#df_weather['just_date'] = df_weather['date'].dt.date nope

print('inspect weather data for Berlin')
print(df_weather.head())

# also save in csv file
df_weather.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/external/weather_data.csv')

####################################################################################################
# Then I load measures data and set the time variable as index to  merge on this later
df_measures = pd.read_excel('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/external/response_graphs_data_2021-06-09.xlsx')
print('Data on measuers', df_measures.head)

# Drop Country as only Germany is included
df_measures.drop(['Country'], axis=1, inplace=True)

# create list and dict for column names
col_mapping_measures = [f"{c[0]}:{c[1]}" for c in enumerate(df_measures.columns)]
col_mapping_dict_measures = {c[0]:c[1] for c in enumerate(df_measures.columns)}
print(col_mapping_dict_measures)
## New Column names for visualization
#col_names = []

# fill in nan in end_date with latest date of  downloaded data: 2021-06-10
# measures with no end date are still in effect, this way they will be coded as yes in the Dummy variable after processing
df_measures.fillna('2021-06-10', inplace=True)
print(df_measures.head())

# reshape data --> one row per day
df_measures["date"] = df_measures.apply(
    lambda x: pd.date_range(x["date_start"], x["date_end"]), axis=1
)
df_m = (
    df_measures.explode("date", ignore_index=True)
    .drop(columns=["date_start", "date_end"])
)

# Datensatz komplett anschauen (150 Zeilen)
pd.set_option("display.max_rows", 150, "display.max_columns", None)
print('what happened to the data?', df_m.head())

# Add column indicating measure is active at given date
# add column of ones (for active measures)
active = np.ones(6355)
df_m['Active'] = active
print('print shape before pivot', df_m.shape)


#print('pivot to allow merging datasets later')
# The active colum of ones is NaN when ther
df_m2=df_m.pivot(index='date', columns='Response_measure')
print('head dfm2', df_m2.head() )

print('print shape after pivot', df_m2.shape)

df_m2.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/response_graphs_pivot.csv')

df_m2 = pd.read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/external/response_graphs_pivot_3.csv', sep=';')

print('see all columns', df_m)

print('df_m2 kopf angepasst', df_m2.head())

# Set date as index (is already index)
#df_m.set_index(df_m['date'], inplace=True)  !! eventuell wieder rein
print('date now index?', df_m.head())

df_m.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/response_graphs_reshaped_date_index.csv')

####################################################################################################
# Third I load covid data and set the time variable as index to  merge on this later
####################################################################################################
# use read_excel to read part of larger dataset more quickly
# df = read_excel('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/external/owid-covid-data.xlsx', nrows=150)
# then use pandas Dataframe
# df = pd.DataFrame(df)
####################################################################################################
#Only Germany (only includes date and numerical data)
df = pd.read_excel('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/external/owid-covid-data-Germany-num.xlsx', sheet_name=0)
#drop information I don't need fo ranalyses

# drop empty columns (no information in these columns for Germany)
print('Head original df', df.shape)
df = df.dropna(axis='columns', how='all')
print('Shape', df.shape)
print('Head after dropping nan columns', df.head())

# fill in weekly infomation with last valid information (use weekly information fo the following week up to the next valid value
##!! not always a good measure! have to choose where to use this method via looking at the information!
df.fillna(method="ffill", inplace=True)
#Fill in the rest of the values with 0 (I have to do this more selectively later on!!)
df.fillna(0, inplace=True)



# Datensatz komplett anschauen
pd.set_option("display.max_rows", None, "display.max_columns", None)
print('see all colums')
print(df.head())

# Column names in dict and list
col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(df.columns)]
col_mapping_dict = {c[0]:c[1] for c in enumerate(df.columns)}

print(col_mapping_dict)

# use datetime to be able to work with dates
#df['date'] = pd.to_numeric(pd.to_datetime(df['date']))
#df['date'] = df[str('date')]

#df['date'] = pd.to_datetime(df['date'])  # , format='%Y%m%d'
# Set date as index
df.set_index(df['date'], inplace=True) #!! eventuell wieder rein

#df_m2.set_index(df_m2['date'], inplace=True)
# Also set index of measurers data to Date
print('Measures data', df_m2.head())
df_m2.set_index('Date', inplace= True)

print('is date now index?')
print(df_m2.head())

#Pivot to allow for joining the datasets


####################################################################################################
# The fourth main step of my data preparation is to join all three data sources together.

#result = pd.concat([df, df_m], axis=1)
#result = pd.concat([df, df_m2, df_weather], axis=1)  # join data on measures with owid data
#result_prep = df.join(df_m2)
#result = result_prep.join(df_weather)
####################################################################################################

# Join datasets
result_prep = df.join(df_m2)  #, how='left', lsuffix='_left', rsuffix='_right')
result = result_prep.join(df_weather)

print('Head merged datasets', result.head())

result.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join4.csv')

####################################################################################################
# For binary analyses I creratet R_kat which is 1 if cases are increasng and 0 if they are decreasing
result['R_kat'] = np.where(result['reproduction_rate'] < 1, 0, 1)
print(result[['R_kat','reproduction_rate']])
# I drop the reproduction rate becaue it is by nature perfectly collinar with R_kat
result.drop(['reproduction_rate'], axis=1, inplace=True)
result.drop(['date'], axis=1, inplace=True)
# rename new index column
df.rename(columns={"Unnamed: 0" : "days since pandemic"}, inplace=True)  ##df.rename(columns={ df.columns[1]: "your value" }, inplace = True)
# I drop some non-numerical data I don't neccassarily need
result.drop(['time'], axis=1, inplace=True)
result.drop(['tests_units'], axis=1, inplace=True)

#Fill in the rest of the values with 0_ this is intuitively useful for the data at hand; without time restrictions I might chosse other procedures


# Fill all remaining missing values with 0 (plausible for first part
result.fillna(0, inplace=True)
result.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join2.csv')

# Für erste Analysen ohne Datumsangabe
result.reset_index(drop=True, inplace=True)

# Drop columns without variance  --> ab jetzt mit df als endgültigem Dataframe arbeiten. Drop if only one unique value
df = pd.DataFrame(result)
print('shape before', df.shape)
df=df[[i for i in df if len(set(df[i]))>1]]
print('shape after', df.shape)

# Other way to achieve this did not work as intended
#Drop inplace
# print('shape before', df.shape)
# for col in df.columns:
#    if len(df[col].unique()) == 1:
#       df.drop(col,inplace=True,axis=1)
# print('shape after', df.shape)

# (I have to do this more selectively later on!!)
df.fillna(0, inplace=True)
print(df.head())




# Liste der Spalten
# Column names in dict and list
#col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(result.columns)]
#col_mapping_dict = {c[0]:c[1] for c in enumerate(result.columns)}
#columns_join1 = result.columns
#print('Column names', columns_join1)  # I use the column names for figures and copy them into visualize.py; more elegant solution coming later

columns_df = df.columns
print('Column names after drop', columns_df)  # I use the column names for figures and copy them into visualize.py; more elegant solution did not work

df.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join1_old.csv')

#Delete features based on logical assumptions about data
#reasoning: vaccinations: total vaccinated and total fully vaccinated per hundred already included;
#excess_mortality: cannot influence infections or death rate
#new_tests_smoothed_per_thousand already in dataset
#weekly_hosp_admissions_per_million in data
df.drop(['new_vaccinations_smoothed_per_million', 'excess_mortality', 'new_vaccinations' ,	'new_vaccinations_smoothed' , 'total_vaccinations_per_hundred', 'total_vaccinations', 'people_vaccinated' , 'people_fully_vaccinated' , 'total_tests' , 'total_tests_per_thousand' ,'new_tests_smoothed', 'weekly_hosp_admissions'], axis=1, inplace=True)
# Also delete
# new_cases: keep only smoothed, total_cases_per_million: keep only total, new_deaths_smoothed_per_million (keep non smoothed)
df.drop(['new_cases', 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million'], axis=1, inplace=True)
#Delete when predicting cases, not if trying to predict deaths! Then keep one for hospitalization, one fo deaths
df.drop(['icu_patients' ,'icu_patients_per_million', 'total_deaths', 'new_deaths' ,	'new_deaths_smoothed', 'total_deaths_per_million', 'new_deaths_per_million'], axis=1, inplace=True)
# Delete more because too many variables, some quiet correlated


columns_df = df.columns
print('Column names after drop', columns_df)  # I use the column names for figures and copy them into visualize.py; more elegant solution coming later

#'überschreibt ursprünglichen Analysedatensatz
df.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join1.csv')

####################################################################################################
# To make use of the time series format I create lead and lag variables that differ for traditional machine learning and the LSTM model
####################################################################################################
# First I prepare data for traditional machine learning
####################################################################################################
# Use lead variable of reproduction rate (equals lagged values of explanatory features)
#End up with dataset that has yesterdays (and up to 5 days in the past) explanatory variables in one row with today's (binary) reproduction rate
leads(data= df, x= df.R_kat, z= 'r_kat_lead_', number=5)  # defined in Preprocessor

#Thisi s how I tried this without the function; delete later
# number_leads = 5
# for lead in range(1, number_leads + 1):
#     df['r_kat_lead_' + str(lead)] = df.R_kat.shift(periods=-lead)

# number_lags = 5
# for lag in range(1, number_lags + 1):
#     df['r_kat_lag_' + str(lag)] = df.R_kat.shift(periods=-lag)

# Drop rows that have missing values now. Those that cannot use lead/lag variables at beginning/end of datsaet
df = df.dropna()

print(df.head())

####################################################################################################
# Here I save the analysis data for the Classification analyses
####################################################################################################

df.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join_lead.csv')

####################################################################################################
#For regression analysese I drop lead caategorical variables and crerate them for the case numbes
####################################################################################################
# Drop lead variables
leadlist = [1, 2, 3, 4, 5]
for i in leadlist:
    df.drop(['r_kat_lead_' + str(i)], axis=1, inplace=True)

# I basically rename new cases smoothed to new cases and put it at the right hand side of the dataset
df['new_cases'] = df['new_cases_smoothed']
df.drop(['new_cases_smoothed'], axis=1, inplace=True)

####################################################################################################
# Here I save data before adding leads which I only use for traditional machine learning
####################################################################################################
df.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join_cases.csv')


# Here I crate up to five leads of new cases, the function is defined in Preprocessor under src/data
leads(data = df, x= df.new_cases, z ='new_cases_lead_', number=5)

####################################################################################################
# Here I save thte analysis data for regression analyses
####################################################################################################
df.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join_lead_cases.csv')


####################################################################################################
####################################################################################################
#Data preparation for Deep Learnng using Pytorch
####################################################################################################
# I start with information without leads and with date as index
df = pd.read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join4.csv')
print('1_how does head look for deep learning', df.head())
df.drop(['tests_units'], axis=1, inplace=True)
df.drop(['reproduction_rate'], axis=1, inplace=True) ##(weil ich sie auch in anderen Analysen nicht verwende)
# Fill all remaining missing values with 0 (plausible for first part
df.fillna(0, inplace=True)
#result.drop(['date'], axis=1, inplace=True) # möglicherweise, weil es auch schon der Index ist.
# Drop columns without variance  --> ab jetzt mit df als endgültigem Dataframe arbeiten. Drop if only one unique value
print('shape before', df.shape)
df=df[[i for i in df if len(set(df[i]))>1]]
print('shape after', df.shape)
df.fillna(0, inplace=True)
print('how does head look for deep learning', df.head())

#weekly_hosp_admissions_per_million in data
df.drop(['new_vaccinations_smoothed_per_million', 'excess_mortality', 'new_vaccinations' ,	'new_vaccinations_smoothed' , 'total_vaccinations_per_hundred', 'total_vaccinations', 'people_vaccinated' , 'people_fully_vaccinated' , 'total_tests' , 'total_tests_per_thousand' ,'new_tests_smoothed', 'weekly_hosp_admissions'], axis=1, inplace=True)
# Also delete
# new_cases: keep only smoothed, total_cases_per_million: keep only total, new_deaths_smoothed_per_million (keep non smoothed)
df.drop(['new_cases', 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million'], axis=1, inplace=True)
#Delete when predicting cases, not if trying to predict deaths! Then keep one for hospitalization, one fo deaths
df.drop(['icu_patients' ,'icu_patients_per_million', 'total_deaths', 'new_deaths' ,	'new_deaths_smoothed', 'total_deaths_per_million', 'new_deaths_per_million'], axis=1, inplace=True)
# Delete more because too many variables, some quiet correlated
# Drop one of the two date variables
#df.drop(df.columns[[1]], axis = 1, inplace = True)
#df = pd.read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join_cases.csv', header=0, index_col=0)

# Set date as index
df.set_index(df['date'], inplace=True)
print('date now index=', df.head())
# drop date in colum
#df.drop(['date', 'time'], axis=1, inplace=True)
print('what type is time?', type(df.time))
time2 = df['time']
print('what type is time2?', type(time2))
print('date still index=', df.head())

# date
#print('Date :',df.index.date()) ##AttributeError: 'Index' object has no attribute 'date'
print('what type is date?', type(df.index))

# Assign time features
df_features = (
                df
                .assign(day = df.index.day)
                .assign(month = df.index.month)
                .assign(day_of_week = df.index.dayofweek)
                .assign(week_of_year = df.index.week)
              )

####################This is where I can't go any further unfortunately
#df = df.set_index(['Datetime'])
#df.index = pd.to_datetime(df.index)
#df.index = df[0]
if not df.index.is_monotonic:
    df = df.sort_index()
    df = df.sort_index()
print('how does head look after setting index?', df.head()) #--> Did not work at all, Philipp fragen, erstmal nicht ins Projekt aufnehmen
#df = df.rename(columns={'PJME_MW': 'value'})
#plot_dataset(df, title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')
# Use easier way to create lags of all variables
from datatable import dt, f, by
DT = dt.Datatable(df)
#DT[, unlist(lapply(.SD, shift, n = 0:3), recursive = FALSE)]

# First I generate time lags for all fetures
def generate_time_lags(df, n_lags):
    feature_list = list(df.columns)
    print(feature_list)
    df_n = df.copy()
    for feat in feature_list:
        for n in range(1, n_lags + 1):
            df_n[f"str(feat) + _lag{n}"] = df_n[str(feat)].shift(n)
        df_n = df_n.iloc[n_lags:]
        return df_n


#input_dim = 100

input_dim = 10
df_generated = generate_time_lags(df, input_dim)
df_generated
print(df_generated.head())
print('did anything happen?')








#############################################################
#Diesen Teil kann man komplett löschen.
#############################################################

#df = pd.DataFrame(df, index=date) #kann weg?
#
# ##First graph??
# print('##First graph??')
# import plotly.graph_objs as go
# from plotly.offline import iplot
#
# def plot_dataset(result, title):
#     data = []
#     value = go.Scatter(
#         x=result.index,
#         y=result.new_deaths,
#         mode="lines",
#         name="new deaths",
#         marker=dict(),
#         text=result.index,
#         line=dict(color="rgba(0,0,0, 0.3)"),
#     )
#     data.append(value)
#
#     layout = dict(
#         title=title,
#         xaxis=dict(title="Date", ticklen=5, zeroline=False),
#         yaxis=dict(title="Value", ticklen=5, zeroline=False),
#     )
#
#     fig = dict(data=data, layout=layout)
#     #iplot(fig)
#     plotly(fig)
#
# plot_dataset(result,title='Title')
#
# #########
# #try using lagged vaiables
#
#
# def generate_time_lags(df, n_lags):
#     df_n = df.copy()
#     for n in range(1, n_lags + 1):
#         df_n[f"lag{n}"] = df_n["value"].shift(n)
#     df_n = df_n.iloc[n_lags:]
#     return df_n
#
#
# input_dim = 100
#
# df_generated = generate_time_lags(df, input_dim)
# df_generated
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# print('Firrst graph??')
#
# import pandas as pd
#
# df = pd.read_csv('<YOUR_FILE_DIR>/PJME_hourly.csv')
#
# df = df.set_index(['Datetime'])
# df.index = pd.to_datetime(df.index)
# if not df.index.is_monotonic:
#     df = df.sort_index()
#
# df = df.rename(columns={'PJME_MW': 'value'})
# plot_dataset(df, title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')
# print('##First graph??')
#
#
#
#
#
#
#
#
#
#
# # creating instance of one-hot-encoder
# # Zunächst reshape
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
# #print('Hier soll onehotlist stehen')
# #print(onehotlist[0][2].categories_)
#
#
# #print(df['location'])
#
# # This works!
# enc_df = pd.DataFrame(onehotlist[0][1].toarray())
# print('Does this work too?')
# enc_df = pd.DataFrame(onehotlist.toarray())
#
# #enc = OneHotEncoder(handle_unknown='ignore')
# # passing cat columns (label encoded values)
# #enc_df = pd.DataFrame(enc.fit_transform(df[['continent', 'location', 'iso_code']]).toarray())
# # merge with main df  on key values
# #df = df.join(enc_df)
# df_join = df.join(pd.DataFrame(enc_df.toarray(), columns=onehot_enc.categories_[0]))
#
# print('did join work?')
# df.set_index('date', inplace=True)  #!! eventuell wieder einkommentieren
# print('did index setting work?')
# print(df.head())
#
#
# print('test')
# #for i in 'continent', 'location', 'iso_code':
# #    df[i] = pd.to_numeric(df[i],errors = 'coerce')
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
# #X = df.loc[:,df.columns != "new_deaths"]
# # use only certain features (leave out barely filled information and information that is not available on a daily basis
#
# X = df.iloc[:, np.r_[0,3:18,29,30]]
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
# #print('Datatype info')
# #print(df.info(verbose=True))
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
# #plot = df.plot(style='.', figsize=(15, 8), title='Entire covid dataset')
# #print(plot)
#
# import pandas as pd
#
# #df = df.set_index(['Datetime'])
# #df.index = pd.to_datetime(df.index)
#
# #if not df.index.is_monotonic:
# #    df = df.sort_index()
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
#
# # Remember for later projects
# To set up project structure using coockie cutter
from cookiecutter.main import cookiecutter
# doch nicht verwendetcookiecutter https://gitlab.com/ericdevost/cookiecutter-python.git
# cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science

# cookiecutter('cookiecutter-pypackage/')

# Plotly to create interactive graphs
