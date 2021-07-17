from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Preprocessor:
    def __init__(self, x, y):
        # hier hab ich daten bekommen

        # hier werden daten gesplittet
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # hier werden die daten an den normalizer übergeben
        self.fit_normalizer(self.X_train)

        # ab hier hab ich jetzt den scaler fertig

    def get_data(self):
        return self.normalize(self.X_train), self.normalize(self.X_test), self.y_train, self.y_test, self.scaler

    # trainiert den normalizer
    def fit_normalizer(self, x):
        # hier kommen die übergebenen daten an
        # hier initialisiert
        self.scaler = MinMaxScaler()

        # hier trainiert
        self.scaler.fit(dims(x))

    def normalize(self, x):
        # wendet einfach den gespeicherten scaler an
        return self.scaler.transform(dims(x))


def dims(x):
    if np.array(x).ndim == 1:
        return np.array(x).reshape(-1, 1)
    else:
        return x


def leads(data, x, z:str, number = 5):
    number_leads = number
    for lead in range(1, number_leads + 1):
            data[z + str(lead)] = x.shift(periods=-lead)

# Frage an Philipp: könnte man sowas umsetzen und falls ja: wie?
class Split:
    def __init__(self, X, y, n):
        self.x = X
        self.y = y
        self.n = n

    def get_xy(self, data, n=6):
        X= data.iloc[:, :-n]
        return self.X
        print('inside function', X)
        y = data.iloc[:, -n]
        return self.y

#def split:

#         X = self.iloc[:, :-n]
#         return X
#         print('inside function', X)
#         y = self.iloc[:, -n]
#         return y
#
#     return self.normalize(self.X_train), self.normalize(self.X_test), self.y_train, self.y_test, self.scaler

#
# def split(self, data, n=6):
#     self.X = data.iloc[:, :-n]
#     return self.X
#     self.y = data.iloc[:, -n]
#     return self.y


