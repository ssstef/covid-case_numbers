
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


df= pd.read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join1.csv')
plot_dataset(df=df, title='First plot')