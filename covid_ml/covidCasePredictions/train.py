import os
import sys
sys.path.append("src/models/")
sys.path.append("src/data/")
import pandas as pd
from Classifier import Classifier



import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from Preprocessor import Preprocessor


from Classifier import Classifier
from operator import itemgetter, attrgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

daten_prepared = pd.read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join1.csv')

print(daten_prepared.head())


X = daten_prepared.loc[:, daten_prepared.columns != "R_kat"]
y = daten_prepared["R_kat"]

X_train, X_test, y_train, y_test, scaler = Preprocessor(X, y).get_data()


# Models spezifieren
models = {"rf": RandomForestClassifier, "dt": DecisionTreeClassifier, "knn": KNeighborsClassifier, "svm": SVC,
          "mlp": MLPClassifier}

# Classifier verwenden
clf = Classifier()
resultat = clf.train_models(X_train, X_test, y_train, y_test, models)

# Bestes Ergebnis bestimmen und als Modell speichern
print("Bestes Model ist: {} mit einer Akkuranz von {}%".format(sorted(resultat, key=itemgetter(1), reverse=True)[0][0],
                                                               sorted(resultat, key=itemgetter(1), reverse=True)[0][
                                                                   1] * 100))
bestes_model = sorted(resultat, key=itemgetter(1), reverse=True)[0][2]
print("Alle Ergebnisse: {}".format(resultat))

