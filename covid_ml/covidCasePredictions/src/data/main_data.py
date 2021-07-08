# Import modules
from datetime import date, datetime

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


print('In this file I analyse covid data')

print('Weather data')
# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
start = datetime(2020, 1, 1)
end = datetime(2021, 7, 1)

# Create Point for Berlin
berlin = Point(50.52008, 13.404954, 34)

# Get daily data for 2018
data = Daily(berlin, start, end)
data = data.fetch()

# Just for fun: Plot line chart including average, minimum and maximum temperature
data.plot(y=['tavg', 'tmin', 'tmax'])
plt.show()
plt.savefig('covid_ml/covidCasePredictions/reports/figures/temperatures_berlin.png')

df_weather = pd.DataFrame(data)
print('inspect weather data for Berlin')
print(df_weather.head)
print('Data on measuers')
df_measures = pd.read_excel('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/external/response_graphs_data_2021-06-09.xlsx')
print(df_measures.head)


# Drop Country as only Germany is included
df_measures.drop(['Country'], axis=1, inplace=True)

# create list and dict for column names
col_mapping_measures = [f"{c[0]}:{c[1]}" for c in enumerate(df_measures.columns)]
col_mapping_dict_measures = {c[0]:c[1] for c in enumerate(df_measures.columns)}
print(col_mapping_dict_measures)


# fill in nan in end_date with latest date of  downloaded data: 2021-06-10
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
print('what happened to the data?')
# Datensatz komplett anschauen (150 Zeilen)
pd.set_option("display.max_rows", 150, "display.max_columns", None)
print(df_m)

# Add column indicating measure is active at given date
# add column of ones (for active measures)
active = np.ones(6355)
df_m['Active'] = active
print('print shape')


#print('pivot to allow merging datasets later')
df_m2=df_m.pivot(index='date', columns='Response_measure')
print(df_m2.shape)



print('print shape')

df_m2.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/response_graphs_pivot.csv')

print('see all columns')
print(df_m)

# Set date as index
df_m.set_index(df_m['date'], inplace=True)
print('date now index?')
print(df_m.head())
df_m.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/response_graphs_reshaped_date_index.csv')

print('In this file I analyse covid data')
# use read_excel to read part of larger dataset more quickly
# Open and inspect data
#All countries
#df = pd.read_excel('covid_ml/covidCasePredictions/data/external/owid-covid-data.xlsx', sheet_name=0)

#Only Germany (only date and numerical data)
df = pd.read_excel('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/external/owid-covid-data-Germany-num.xlsx', sheet_name=0)
# drop empty columns (no information for Germany)
print(df.shape)
df = df.dropna(axis='columns', how='all')
print(df.shape)
print(df.head())
print('Columns dropped?')
# fill in weekly infomation with last valid information (use weekly information fo the following week up to the next valid value
##!! not always a good measure! have to choose where to use this method via looking at the information!
df.fillna(method="ffill", inplace=True)
#Fill in the rest of the values with 0 (I have to do this more selectively later on!!)
df.fillna(0, inplace=True)
#use subset of data
#Quicker way to read subset of large dataset
#df = read_excel('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/external/owid-covid-data.xlsx', nrows=150)

#df = read_excel('covid_ml/covidCasePredictions/data/external/owid-covid-data.xlsx', nrows=150)
# use pandas Dataframe
#df = pd.DataFrame(df)

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
df.set_index(df['date'], inplace=True)
print('is date now index?')
print(df.head())

#Pivot to allow for joining the datasets


# Join datasets
#result = pd.concat([df, df_m], axis=1)
result = pd.concat([df, df_m2, df_weather], axis=1)  # join data on measures with owid data
print('did join work?')
# result = pd.concat([df, df_m2], axis=1) #

print(result.head())

result['R_kat'] = np.where(result['reproduction_rate'] < 1, 0, 1)
print(result[['R_kat','reproduction_rate']])

result.drop(['reproduction_rate'], axis=1, inplace=True)
result.drop(['date'], axis=1, inplace=True)
result.drop(['tests_units'], axis=1, inplace=True)
#result['R_kat'] = 0 if result['reproduction_rate'] < 1 else 1
#result['R_kat'] = result['reproduction_rate'>=1==1, 'reproduction_rate'<1 ==0]
#Fill in the rest of the values with 0 (I have to do this more selectively later on!!)
df.fillna(0, inplace=True)

print(df.head())
#Also save data in csv®

# Für erste Analysen ohne Datumsangabe
result.reset_index(drop=True, inplace=True)


result.fillna(0, inplace=True)
result.to_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join1.csv')
#df = pd.DataFrame(df, index=date) #kann weg?

##First graph??
print('##First graph??')
import plotly.graph_objs as go
from plotly.offline import iplot

def plot_dataset(result, title):
    data = []
    value = go.Scatter(
        x=result.index,
        y=result.new_deaths,
        mode="lines",
        name="new deaths",
        marker=dict(),
        text=result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    #iplot(fig)
    plotly(fig)

plot_dataset(result,title='Title')

#########
#try using lagged vaiables


def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n


input_dim = 100

df_generated = generate_time_lags(df, input_dim)
df_generated
















print('Firrst graph??')

import pandas as pd

df = pd.read_csv('<YOUR_FILE_DIR>/PJME_hourly.csv')

df = df.set_index(['Datetime'])
df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()

df = df.rename(columns={'PJME_MW': 'value'})
plot_dataset(df, title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')
print('##First graph??')










# creating instance of one-hot-encoder
# Zunächst reshape

categorical = ['location', 'iso_code', 'continent']
onehotlist= list()

for i in categorical:
    onehot_enc = OneHotEncoder()
    i_spalte = df[i]
    i_spalte_res = i_spalte.values.reshape(-1,1)
    i_onehot = onehot_enc.fit_transform(i_spalte_res)
    onehotlist.append([i, i_onehot, onehot_enc])
#print('Hier soll onehotlist stehen')
#print(onehotlist[0][2].categories_)


#print(df['location'])

# This works!
enc_df = pd.DataFrame(onehotlist[0][1].toarray())
print('Does this work too?')
enc_df = pd.DataFrame(onehotlist.toarray())

#enc = OneHotEncoder(handle_unknown='ignore')
# passing cat columns (label encoded values)
#enc_df = pd.DataFrame(enc.fit_transform(df[['continent', 'location', 'iso_code']]).toarray())
# merge with main df  on key values
#df = df.join(enc_df)
df_join = df.join(pd.DataFrame(enc_df.toarray(), columns=onehot_enc.categories_[0]))

print('did join work?')
df.set_index('date')
print('did index setting work?')
print(df.head())


print('test')
#for i in 'continent', 'location', 'iso_code':
#    df[i] = pd.to_numeric(df[i],errors = 'coerce')

print('is there categorical info left?')
print(df.info())

#Delete rows with missing data in relevant features
df.dropna(how='any',
          subset=['new_deaths', 'hosp_patients_per_million', 'new_vaccinations'])



print(df.head())
#X = df.loc[:,df.columns != "new_deaths"]
# use only certain features (leave out barely filled information and information that is not available on a daily basis

X = df.iloc[:, np.r_[0,3:18,29,30]]
y = df["new_deaths"]



# Missing Values in den Daten?
print('null values in data')
print(df.isnull().any(axis=1).sum())

print('na values in data')
print(df.isna().any(axis=1).sum())

# print("Original Datafame: ")
pd.set_option('display.max_rows', None, 'display.max_columns', 1000)

print(df.head())
print(df.shape)
#Datatype info
#print('Datatype info')
#print(df.info(verbose=True))

print(df.describe())
('Showing entire dataframe?')
#print(df.describe(percentiles=[.1, .99]))
#print(df.describe(include='all'))

# Encoding of categorical data

# Keep only rows without missing data
# Drop rows which contain any NaN values --> empty data frame --> choose which infomation is necessary
# mod_df = df.dropna()

# Drop rows which contain any NaN value in the selected columns
# mod_df = df.dropna(how='any',
#                 subset=['new_deaths', 'hosp_patients_per_million', 'new_vaccinations'])

# print("Modified Dataframe : ")
# print(mod_df.head())


# Data preprocessing
#X = df.loc[:, df.columns != 'new_deaths']
# try out with subse tof data
X = df.loc[:, df.columns != "new_deaths"]
y = df['new_deaths']



from covid_ml.covidCasePredictions.src.data.Preprocessor import Preprocessor
print('test new X')
X_train, X_test, y_train, y_test = Preprocessor(X, y).get_data()
print('using Preprocessor')


# Descirptive analyses
# Plot the data

import plotly.graph_objs as go
from plotly.offline import iplot

#plot = df.plot(style='.', figsize=(15, 8), title='Entire covid dataset')
#print(plot)

import pandas as pd

#df = df.set_index(['Datetime'])
#df.index = pd.to_datetime(df.index)

#if not df.index.is_monotonic:
#    df = df.sort_index()

# df = df.rename(columns={'PJME_MW': 'value'})
#plot_dataset(df, title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')

# _ = df['new_deaths'].plot.hist(figsize=(15, 5), bins=200, title='Distribution of New Deaths')

# plot_dataset(df, title='covid', xvalue="date", yvalue="new_deaths")
# Models

# Figures

# Remember for later projects
# To set up project structure using coockie cutter
from cookiecutter.main import cookiecutter
# doch nicht verwendetcookiecutter https://gitlab.com/ericdevost/cookiecutter-python.git
# cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science

# cookiecutter('cookiecutter-pypackage/')

# Plotly to create interactive graphs
