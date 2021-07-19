
import plotly.graph_objs as go
from plotly.offline import iplot
import pandas as pd
import matplotlib.pyplot as plt

def plot_dataset(df, title):
    data = []
    R_kat = go.Scatter(
        x=df.index,
        y=df.R_kat,
        mode="lines",
        name="values",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(R_kat)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="R_kat", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)
    #plt.show(fig)


df= pd.read_csv('/covid_ml/covidCasePredictions/data/processed/join1.csv')
plot_dataset(df=df, title='First plot')

import pandas as pd

df = pd.read_csv('/covid_ml/covidCasePredictions/data/processed/join1.csv')

#df = df.set_index(['Datetime'])
#df.index = pd.to_datetime(df.index)
#if not df.index.is_monotonic:
#    df = df.sort_index()

df = df.rename(columns={'Covid': 'R_kat'})
plot_dataset(df, title='Second plot')


#Generatting Ttime lags
def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n


input_dim = 100

df_generated = generate_time_lags(df, input_dim)
df_generated