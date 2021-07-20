import os
import sys

from covid_ml.covidCasePredictions.src.visualization.visualize import decision_tree

sys.path.append('visualization')
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import visualize
from visualize import decision_tree_reg
from sklearn import tree
from sklearn.tree import export_graphviz
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

import numpy as np
import pandas as pd
import pickle as pi

# Heatmap
import matplotlib.pyplot as plt


#cross_val_score(regressor, X, y, cv=10)

class Regressor:
    def __init__(self):
        # Array für alle Ergebnisse
        self.ergebnis = []

    def train_models(self, X_train, X_test, y_train, y_test, models):
        for self.model in models:
        # -----------------------
        # KNN
        # -----------------------
            if self.model == 'knn':
            # Optimalen Knn-Classifier bestimmen
                error = []
                for i in range(1, 40):
                    knn = KNeighborsRegressor(n_neighbors=i)
                    knn.fit(X_train, y_train)
                    pred_i = knn.predict(X_test)  #print(knn.predict([[1.5]]))
                    error.append(np.mean(pred_i != y_test))
                    #error.append(score(X_test, y_test, sample_weight=None))

                 # Knn-Classifier trainieren
                knnreg = KNeighborsRegressor(n_neighbors=7)
                knnreg.fit(X_train, y_train)

                # Knn-Classifier Akkuranz bestimmen
                score = knnreg.score(X_test, y_test)
                self.ergebnis.append(['knn-regressor', score, knn])

            # -----------------------
            # Decision Tree
            # -----------------------
            elif self.model == 'dt':
                # class_weight gebrauchen für DT und RF

                # Optimalen Decision Tree bestimmen
                # Zu testende Decision Tree Parameter
                dt_reg = DecisionTreeRegressor(criterion ='mse', max_depth=5, min_samples_split = 5)
                #tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': [i for i in range(1, 20)],
                 #            'min_samples_split': [i for i in range(2, 20)]}

                # GridSearchCV
                #grd_reg = GridSearchCV(dt_reg, tree_para, cv=5)
                #grd_reg.fit(X_train, y_train)
                print('Regression Decision Tree ausführen')

                # Besten gefundenen Decision Tree übergeben
                #dt_reg = grd_reg.best_estimator_

                # Fit decision tree
                dt_reg.fit(X_train, y_train)



                # Besten gefundenen Decision Tree ausgeben
                decision_tree_reg(best_model=dt_reg, X=X_test)
                print('Regression Decision Tree Ausgabe beendet')

                # Decision Tree ausgeben (wenn Zeit besten Tree suchen und den ausgeben!=
               # decision_tree_reg(best_model=dt, X=X_test) auskommentiert weil der unveränderte Befehl plötzlich Fehler ausgiebt:ValueError: Length of feature_names, 56 does not match number of features, 57 Ich füge eins hinzu: ValueError: Length of feature_names, 58 does not match number of features, 57)

                score = dt_reg.score(X_test, y_test)
                self.ergebnis.append(['decision tree', score, dt_reg])

            # -----------------------

            # -----------------------
            # Random Forest
            # -----------------------
            elif self.model == 'rf':

           # rf = RandomForestClassifier(max_depth=8, criterion="entropy", min_samples_split=9)
                rf = RandomForestRegressor(criterion="mse", max_depth=10, n_estimators=100, random_state=14)
                #forrest_para = {'criterion': ['gini', 'entropy'], 'max_depth': [i for i in range(1, 20)],
                 #            'min_samples_split': [i for i in range(2, 20)]}

                rf.fit(X_train, y_train)

                # GridSearchCV (takes too long!)
                #grd_rf = GridSearchCV(rf, forrest_para, cv=5)
                #grd_rf.fit(X_train, y_train)

                # Besten gefundenen RF übergeben
                #rf_reg = grd_rf.best_estimator_

                score = rf.score(X_test, y_test)
                self.ergebnis.append(['random forest', score, rf])  # rf_reg, davor rf

            # -----------------------
            # Gradient Boosting Regressor
            # -----------------------
            elif self.model == 'gradient':
                gradient = GradientBoostingRegressor(random_state=0)
                gradient.fit(X_train, y_train)
                GradientBoostingRegressor(random_state=0)
               # gradient.predict()

                score = gradient.score(X_test, y_test)
                self.ergebnis.append(['gradient boosting', score, gradient])  # rf_reg, davor rf



                # -----------------------
                # Support Vector Machine
                # -----------------------
            elif self.model == 'svm':
                regr = svm.SVR(kernel='poly')
                regr.fit(X_train, y_train)
                score = regr.score(X_test, y_test)
                self.ergebnis.append(['support vector machine', score, regr])


                # -----------------------
                # Ensemble Learning (Ada Boost)
                # -----------------------
            elif self.model == 'ada':
                ada = AdaBoostRegressor(n_estimators=100, random_state=14)
                ada.fit(X_train, y_train)
                scores = cross_val_score(ada, X_test, y_test, cv=5)
                #score = scores.mean()
                score = ada.score(X_test, y_test)
                self.ergebnis.append(['Ensemble Learning (Ada Boost)', score, ada])


            # -----------------------
            # Ensemble Learning (Voting)
            # -----------------------
            elif self.model == 'voting':
                r1 = LinearRegression(random_state=1)
                r2 = RandomForestRegressor(n_estimators=50, random_state=1)
                er = VotingRegressor([('lr', r1), ('rf', r2)])
                er.fit(X_train, y_train)
                score = er.score(X_train, y_train)


                # print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
                self.ergebnis.append(['Enseble Learning - Voting', score, er])




            # -----------------------
            # MLP
            # -----------------------
            elif self.model == 'mlp':
                mlp = MLPRegressor(hidden_layer_sizes=[100, 100], max_iter=4000, solver='sgd'
                                    , learning_rate='adaptive', learning_rate_init=0.01, n_iter_no_change=200,
                                    early_stopping=True)
                mlp.fit(X_train, y_train)
                mlp.predict(X_test[:])
                score = mlp.score(X_test, y_test)
                self.ergebnis.append(['multi-layer perceptron', score, mlp])
                print("iterations: {}; layers: {}; loss: {}".format(mlp.n_iter_, mlp.n_layers_, mlp.loss_))
                epochs = np.linspace(1, mlp.n_iter_, mlp.n_iter_)

                # plt.plot(epochs, mlp.loss_curve_, label="Fehlerfunktion")
                # plt.plot(weight,2* weight,label="Ableitung")
                # plt.show()

                return self.ergebnis



