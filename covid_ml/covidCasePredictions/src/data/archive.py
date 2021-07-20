# Old ideas and approaches that did not work the way I intended them to.
#####################################################################################################
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
