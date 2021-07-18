import os
import sys
sys.path.append("src/models/")
sys.path.append("src/data/")
sys.path.append("src/visualization/")
import pandas as pd

#SGD and Ensemble
import scipy.stats as stats
from sklearn.utils.fixes import loguniform

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
# lokal gespeichert/selbst geschrieben
#import Preprocessor
from Preprocessor import Preprocessor
#from Preprocessor import Split  #delete later if not possible!
#from Preprocessor import split #delete later!
from Classifier import Classifier
import Regressor
from Regressor import Regressor
from visualize import decision_tree
#from visualize import confmat_plot
from Classifier import correlation
from Classifier import high_correlation

from operator import itemgetter, attrgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#import matplotlib.pyplot as plt
#from sklearn.metrics import plot_confusion_matrix

# Regressors
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import visualize
from visualize import decision_tree
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
# Voting

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

# Neural Network
from sklearn.neural_network import MLPRegressor


################################################################################################################
# Load data, name output categories
df = pd.read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join_lead.csv')  #join1 klappt !!
class_names = ['decreasing covid cases', 'increasing covid cases']

# Save correlation matrix
#high_correlation(data=df)  # delete later (didn't really do anything)
#correlation(data=df)

print(df.head())

################################################################################################################
# Split data into features(X) and target(y)
# put in 1 to predict today's values, 2 for tomorrow's...up to 6 for 5 days after current day
#split(data=df, n=6)
# function not possible so:
# Change number tto 1 to predict today's case numbers, 2 for tomorrow's and so on up to 5 for 5 days in the future
number = int(6)
X = df.iloc[:, :-number]
y = df.iloc[:, -number]



# To use R_kat -->  predict today's reproduction rate
#
# X = df.iloc[:, :-6]
# y = df.iloc[:, -6]
# # To predict reproduction rate 5 days from now
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
#y = df[:, 'r_kat_lag_5']


X_train, X_test, y_train, y_test, scaler = Preprocessor(X, y).get_data()

print('Preprocessor worked?')

# Models spezifieren
models = {"rf": RandomForestClassifier, "dt": DecisionTreeClassifier, "knn": KNeighborsClassifier, "svm": SVC,
          "mlp": MLPClassifier, "ada": AdaBoostClassifier, } #"voting": VotingClassifier

# Classifier verwenden
# clf = Classifier()
# resultat = clf.train_models(X_train, X_test, y_train, y_test, models)

# Bestes Ergebnis bestimmen und als Modell speichern
# print("Bestes Model ist: {} mit einer Akkuranz von {}%".format(sorted(resultat, key=itemgetter(1), reverse=True)[0][0],
#                                                                sorted(resultat, key=itemgetter(1), reverse=True)[0][
#                                                                    1] * 100))
# bestes_model = sorted(resultat, key=itemgetter(1), reverse=True)[0][2]
# print("Alle Ergebnisse: {}".format(resultat))

#Confusion Matrix für das beste Modell ausgeben
np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# for model in models:
#     titles_options = [("Confusion matrix, without normalization", None),
#                   ("Normalized confusion matrix", 'true')]
#     for title, normalize in titles_options:
#         disp = plot_confusion_matrix(model, X_test, y_test,
#                                 display_labels=class_names,
#                                 cmap=plt.cm.Blues,
#                                 normalize=normalize)
#         disp.ax_.set_title(title)
#         print(title)
#         print(disp.confusion_matrix)
#         plt.show()
#         plt_name = "Confmat" + model + ".png"
#         plt.savefig(Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/reports/figures/plt_name)
#

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



#Print confusion matrix
#confmat_plot(cm, alg='dt')


# specify parameters and distributions to sample from
param_dist = {'average': [True, False],
              'l1_ratio': stats.uniform(0, 1),
              'alpha': loguniform(1e-4, 1e0)}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(X_test, y_test)
#print("RandomizedSearchCV took %.2f seconds for %d candidates"
 #     " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


#### Train regresssion
# load dataset
df = pd.read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join_lead_cases.csv')  #join1 klappt !!
#dataset = pd.read_csv('/Users/stefanieunger/PycharmProjects/covid-case_numbers/covid_ml/covidCasePredictions/data/processed/join_lead_cases.csv', header=0, index_col=0)
################################################################################################################
# Change number tto 1 to predict today's case numbers, 2 for tomorrow's and so on up to 5 for 5 days in the future
# number = int(6) (defined above)
X = df.iloc[:, :-number]
y = df.iloc[:, -number]

################################################################################################################

X_train, X_test, y_train, y_test, scaler = Preprocessor(X, y).get_data()

# Models spezifieren
models = {"rf": RandomForestRegressor, "dt": DecisionTreeRegressor, "knn": KNeighborsRegressor, "svm": SVC,
          "mlp": MLPRegressor, "ada": AdaBoostClassifier, 'voting': VotingRegressor } #"voting": VotingClassifier

# Regressor verwenden

reg = Regressor()
resultat = reg.train_models(X_train, X_test, y_train, y_test, models)

# Bestes Ergebnis bestimmen und als Modell speichern
print("Bestes Model ist: {} mit einem R-Quadrat von {}".format(sorted(resultat, key=itemgetter(1), reverse=True)[0][0],
                                                               sorted(resultat, key=itemgetter(1), reverse=True)[0][
                                                                   1] ))
bestes_model = sorted(resultat, key=itemgetter(1), reverse=True)[0][2]
print("Alle Ergebnisse: {}".format(resultat))