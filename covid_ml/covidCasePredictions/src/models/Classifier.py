from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import pickle as pi


class Classifier:
    def __init__(self):
        # Array für alle Ergebnisse
        self.ergebnis = []

    def train_models(self, X_train, X_test, y_train, y_test, models):
        for self.model in models:
            # -----------------------
            # Knn-Classifier
            # -----------------------
            if self.model == 'knn':
                # Optimalen Knn-Classifier bestimmen
                error = []
                for i in range(1, 40):
                    knn = KNeighborsClassifier(n_neighbors=i)
                    knn.fit(X_train, y_train)
                    pred_i = knn.predict(X_test)
                    error.append(np.mean(pred_i != y_test))

                # Knn-Classifier trainieren
                knnclf = KNeighborsClassifier(n_neighbors=7)
                knnclf.fit(X_train, y_train)

                # Knn-Classifier Akkuranz bestimmen
                score = knnclf.score(X_test, y_test)
                self.ergebnis.append(['knn-classifier', score, knnclf])
            # -----------------------

            # -----------------------
            # Decision Tree
            # -----------------------
            elif self.model == 'dt':
                # class_weight gebrauchen für DT und RF

                # Optimalen Decision Tree bestimmen
                # Zu testende Decision Tree Parameter
                dt = DecisionTreeClassifier()
                tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': [i for i in range(1, 20)],
                            'min_samples_split': [i for i in range(2, 20)]}

                #tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': [i for i in range(1, 3)],
                 #           'min_samples_split': [i for i in range(2, 20)]}

                # GridSearchCV
                grd_clf = GridSearchCV(dt, tree_para, cv=5)
                grd_clf.fit(X_train, y_train)

                # Besten gefundenen Decision Tree übergeben
                dt_clf = grd_clf.best_estimator_

                score = dt_clf.score(X_test, y_test)
                self.ergebnis.append(['decision tree', score, dt_clf])
            # -----------------------

            # -----------------------
            # Random Forest
            # -----------------------
            elif self.model == 'rf':
                # rf = RandomForestClassifier(max_depth=8, criterion="entropy", min_samples_split=9)
                rf = RandomForestClassifier(n_estimators=100)
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                self.ergebnis.append(['random forest', score, rf])
            # -----------------------

            # -----------------------
            # Support Vector Machine
            # -----------------------
            elif self.model == 'svm':
                svm = SVC(kernel='poly')
                svm.fit(X_train, y_train)
                score = svm.score(X_test, y_test)
                self.ergebnis.append(['support vector machine', score, svm])

            # -----------------------
            # MLP
            # -----------------------
            elif self.model == 'mlp':
                mlp = MLPClassifier(hidden_layer_sizes=[100, 100], max_iter=5000, solver='sgd'
                                    , learning_rate='adaptive', learning_rate_init=0.01, n_iter_no_change=200,
                                    early_stopping=True)
                mlp.fit(X_train, y_train)
                score = mlp.score(X_test, y_test)
                self.ergebnis.append(['multi-layer perceptron', score, mlp])
                print("iterations: {}; layers: {}; loss: {}".format(mlp.n_iter_, mlp.n_layers_, mlp.loss_))
                epochs = np.linspace(1, mlp.n_iter_, mlp.n_iter_)

                # plt.plot(epochs, mlp.loss_curve_, label="Fehlerfunktion")
                # plt.plot(weight,2* weight,label="Ableitung")
                # plt.show()

        return self.ergebnis