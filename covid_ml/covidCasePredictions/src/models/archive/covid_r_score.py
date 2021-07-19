import sys

import pandas as pd
import pickle as pi
from sklearn.svm import SVC

# working directory: /home/farzaneh/DataScientist/Projekt/scheidung/models/Divorce.py

current_dir = 'covid-case-numbers'
sys.path.append('../covid_ml/covidCasePredictions/src/features/')
from build_features import *

sys.path.append('../covid_ml/covidCasePredictions/src/models/')
from covid_ml.covidCasePredictions.src.models.archive import train_model

sys.path.append('../covid_ml/covidCasePredictions/src/visualization/')
from visualize import *


# https://stackabuse.com/creating-and-importing-modules-in-python/

# reading Data

data = pd.DataFrame('covid_ml/covidCasePredictions/data/processed/join1.csv')

# preparing Data for modelling
preprocessor = Preprocessor(data)
'''
Akkuranz = []
for i in range (2,9):
    test_size = 0.1*i
    X_train, X_test, y_train, y_test, X, y = preprocessor.get_data(test_size)
    #Data Statistics
    count_positive, count_negative, percentage_positive, percentage_negative = preprocessor.statistica()
    #choosing the model
    algo="DT" # algo = ["DT","rf","knn","svm"]
    #training the model
    trained_model, model_confusion_mat = train_model(X_train, X_test, y_train, y_test, algo)
    #model optimisation
    if algo == "svm":
        best_model_parameters = SVC(C=0.1, gamma='auto', kernel='poly', probability=True)
    else:
        best_model_parameters = optimize_model(trained_model, X_train, y_train, algo)
    best_model, best_model_confusion_mat, mean_accuracy_best_parameters = kfoldevaluate_optimzed_model(algo, best_model_parameters, X, y, X_train, X_test, y_train, y_test)
    Akkuranz.append(round(mean_accuracy_best_parameters,4)*100)
#Akkuranz plot
Akkuranz_pl(Akkuranz, algo)
'''
# Either the top one is commented or the one bellow, above is a search for the best train/test size and bellow is modelling with the best calculated variables, with all the different methods
i = 5
test_size = 0.1 * i
X_train, X_test, y_train, y_test, X, y = preprocessor.get_data(test_size)

# Data Statistics
count_positive, count_negative, percentage_positive, percentage_negative = preprocessor.statistica()

print('Cases who did return their credit in percent: {}%'.format(round(percentage_positive, 3)))  # class one
print('Cases who did not return their credit in percent: {}%'.format(round(percentage_negative, 4)))  # class zero

# choosing the model
algorithms = ["DT", "rf", "knn", "svm"]
trained_model_list = []
model_confusion_mat_list = []
best_model_parameters_list = []
best_model_list = []
best_model_confusion_mat_list = []
mean_accuracy_best_parameters_list = []
fpr_list = []
tpr_list = []
threshold_list = []
roc_auc_list = []

for i in range(0, 3):  # 4 is for svm
    algo = algorithms[i]

    # training the model
    trained_model, model_confusion_mat = train_model(X_train, X_test, y_train, y_test, algo)

    trained_model_list.append(trained_model)
    # model_confusion_mat_list.append(model_confusion_mat)

    # model optimisation
    if algo == "svm":
        best_model_parameters = SVC(C=1.0, gamma='auto', kernel='linear', probability=True)
    else:
        best_model_parameters = optimize_model(trained_model, X_train, y_train, algo)

    best_model_parameters_list.append(best_model_parameters)

    best_model, best_model_confusion_mat, mean_accuracy_best_parameters = kfoldevaluate_optimzed_model(algo,
                                                                                                       best_model_parameters,
                                                                                                       X, y, X_train,
                                                                                                       X_test, y_train,
                                                                                                       y_test)
    best_model_list.append(best_model)
    best_model_confusion_mat_list.append(best_model_confusion_mat)
    mean_accuracy_best_parameters_list.append(mean_accuracy_best_parameters)

    # ROC calculation
    fpr, tpr, threshold, roc_auc = roc_vorbereitung(best_model, X_test, y_test)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    threshold_list.append(threshold)
    roc_auc_list.append(roc_auc)

    # graphical output
    pl_ROC(algo, roc_auc, fpr, tpr)
    if algo == "DT": Diabetes_tree(best_model, X)

    # save in Pickle file
    file_name = "classification_model" + algo + ".pickle"
    fill = open(file_name, 'wb')  # allow to Write the file in a Binary format
    pi.dump(best_model, fill)
    fill.close()

ConfmatPl(best_model_confusion_mat_list, algorithms[0:4], vote_method)
