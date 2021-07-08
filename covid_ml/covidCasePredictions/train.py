import os
import sys
sys.path.append("src/models/")
sys.path.append("src/data/")
import pandas as pd
from Classifier import Classifier

#SGD and Ensemble
import scipy.stats as stats
from sklearn.utils.fixes import loguniform

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from Preprocessor import Preprocessor


from Classifier import Classifier
from operator import itemgetter, attrgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

daten_prepared = pd.read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join1.csv')  #join1 klappt !!

print(daten_prepared.head())


X = daten_prepared.loc[:, daten_prepared.columns != "R_kat"]
y = daten_prepared["R_kat"]

X_train, X_test, y_train, y_test, scaler = Preprocessor(X, y).get_data()


# Models spezifieren
models = {"rf": RandomForestClassifier, "dt": DecisionTreeClassifier, "knn": KNeighborsClassifier, "svm": SVC,
          "mlp": MLPClassifier, "ada": AdaBoostClassifier, } #"voting": VotingClassifier

# Classifier verwenden
clf = Classifier()
resultat = clf.train_models(X_train, X_test, y_train, y_test, models)

# Bestes Ergebnis bestimmen und als Modell speichern
print("Bestes Model ist: {} mit einer Akkuranz von {}%".format(sorted(resultat, key=itemgetter(1), reverse=True)[0][0],
                                                               sorted(resultat, key=itemgetter(1), reverse=True)[0][
                                                                   1] * 100))
bestes_model = sorted(resultat, key=itemgetter(1), reverse=True)[0][2]
print("Alle Ergebnisse: {}".format(resultat))


# Parameter optimieren
# build a SGD classifier
clf = SGDClassifier(loss='hinge', penalty='elasticnet',
                    fit_intercept=True)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {'average': [True, False],
              'l1_ratio': stats.uniform(0, 1),
              'alpha': loguniform(1e-4, 1e0)}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(X_test, y_test)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)