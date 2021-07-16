import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import sys, os
import seaborn as sns

sys.path.append("..")


def pl_roc(algo, roc_auc, fpr, tpr):
    # change the working directory
    path_start = os.getcwd()
    pathr = os.path.dirname(os.getcwd()) + '/covidCasePredictions/reports/figures'
    os.chdir(pathr)

    # define figure size
    plt.figure(figsize=(12, 12))

    # add title
    plt.title('{} ROC Curve'.format(algo))

    # plot and add labels to plot
    plt.plot(fpr, tpr, 'b', label='Covid data: {} AUC =  {}'.format(algo, (round(roc_auc, 4))))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt_name = "ROC_Curve_" + algo + ".png"
    plt.savefig(plt_name)

    # plt.show()

    # change to the start working directory
    os.chdir(path_start)
    return


def decision_tree(best_model, X):
    plt.figure(figsize=(30, 14))
    tree.plot_tree(best_model, filled=True, fontsize=12)

    # change the working directory
    path_start = os.getcwd()
    pathr = os.path.dirname(os.getcwd()) + '/covidCasePredictions/reports/figures'
    os.chdir(pathr)
    #first get colnames
    #x_columns = pd.DataFrame(X)
    #columns = x_columns.columns
    # export_graphviz(best_model, out_file=("Covid_tree.dot"),  feature_names= columns,
    #                 class_names=(['decreasing case numbers', 'increasing case numbes']), rounded=True, filled=True)
    # os.system("dot -Tpng Covid_tree.dot -o Covid_tree.png")
    # os.system("dot -Tps Covid_tree.dot -o Covid_tree.ps")

   # export_graphviz(best_model, out_file=("Covid_tree.dot"), feature_names=X.columns[:],
    export_graphviz(best_model, out_file=("Covid_tree.dot"),
                    feature_names=['day since pandemic', 'total_cases', 'new_cases_smoothed',
                               'weekly_hosp_admissions_per_million', 'new_tests_smoothed_per_thousand',
                               'positive_rate', 'tests_per_case', 'people_vaccinated_per_hundred',
                               'people_fully_vaccinated_per_hundred', 'stringency_index',
                               'BanOnAllEvents', 'BanOnAllEventsPartial', 'ClosDaycare',
                               'ClosDaycarePartial', 'ClosPrim', 'ClosPrimPartial', 'ClosPubAny',
                               'ClosPubAnyPartial', 'ClosSec', 'ClosSecPartial', 'EntertainmentVenues',
                               'EntertainmentVenuesPartial', 'GymsSportsCentres',
                               'GymsSportsCentresPartial', 'HotelsOtherAccommodationPartial',
                               'IndoorOver100', 'IndoorOver1000', 'IndoorOver50',
                               'MasksMandatoryAllSpacesPartial', 'MasksMandatoryClosedSpaces',
                               'MasksVoluntaryAllSpaces', 'MassGather50', 'MassGather50Partial',
                               'MassGatherAll', 'MassGatherAllPartial', 'NonEssentialShops',
                               'NonEssentialShopsPartial', 'OutdoorOver100', 'PlaceOfWorshipPartial',
                               'PrivateGatheringRestrictions', 'QuarantineForInternationalTravellers',
                               'QuarantineForInternationalTravellersPartial', 'RestaurantsCafes',
                               'RestaurantsCafesPartial', 'SocialCirclePartial', 'StayHomeGen',
                               'StayHomeGenPartial', 'StayHomeOrderPartial', 'Teleworking',
                               'TeleworkingPartial', 'tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd',
                               'wpgt', 'pres'],
                class_names=(['decreasing case numbers', 'increasing case numbes']), rounded=True, filled=True)
    os.system("dot -Tpng Covid_tree.dot -o Covid_tree.png")
    os.system("dot -Tps Covid_tree.dot -o Covid_tree.ps")

    # change to the start working directory
    os.chdir(path_start)
    return

    # export_graphviz(best_model, out_file=("Covid_tree.dot"),  feature_names=['day since pandemic' , 'total_cases', 'new_cases', 'new_cases_smoothed',
    #    'total_cases_per_million', 'new_cases_per_million',
    #    'new_cases_smoothed_per_million', 'total_deaths_per_million',
    #    'new_deaths_per_million', 'new_deaths_smoothed_per_million',
    #    'weekly_hosp_admissions_per_million', 'new_tests_smoothed_per_thousand',
    #    'positive_rate', 'tests_per_case', 'people_vaccinated_per_hundred',
    #    'people_fully_vaccinated_per_hundred', 'stringency_index',
    #    'BanOnAllEvents', 'BanOnAllEventsPartial', 'ClosDaycare',
    #    'ClosDaycarePartial', 'ClosPrim', 'ClosPrimPartial', 'ClosPubAny',
    #    'ClosPubAnyPartial', 'ClosSec', 'ClosSecPartial', 'EntertainmentVenues',
    #    'EntertainmentVenuesPartial', 'GymsSportsCentres',
    #    'GymsSportsCentresPartial', 'HotelsOtherAccommodationPartial',
    #    'IndoorOver100', 'IndoorOver1000', 'IndoorOver50',
    #    'MasksMandatoryAllSpacesPartial', 'MasksMandatoryClosedSpaces',
    #    'MasksVoluntaryAllSpaces', 'MassGather50', 'MassGather50Partial',
    #    'MassGatherAll', 'MassGatherAllPartial', 'NonEssentialShops',
    #    'NonEssentialShopsPartial', 'OutdoorOver100', 'PlaceOfWorshipPartial',
    #    'PrivateGatheringRestrictions', 'QuarantineForInternationalTravellers',
    #    'QuarantineForInternationalTravellersPartial', 'RestaurantsCafes',
    #    'RestaurantsCafesPartial', 'SocialCirclePartial', 'StayHomeGen',
    #    'StayHomeGenPartial', 'StayHomeOrderPartial', 'Teleworking',
    #    'TeleworkingPartial', 'tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd',
    #    'wpgt', 'pres'],
    #                 class_names=(['decreasing case numbers', 'increasing case numbes']), rounded=True, filled=True)
    # os.system("dot -Tpng Covid_tree.dot -o Covid_tree.png")
    # os.system("dot -Tps Covid_tree.dot -o Covid_tree.ps")
    #
    # # change to the start working directory
    # os.chdir(path_start)
    # return


# #Old Version
#     export_graphviz(best_model, out_file=("Covid_tree.dot"),  feature_names=['day since pandemic' , 'total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths',
#        'new_deaths', 'new_deaths_smoothed', 'total_cases_per_million',
#        'new_cases_per_million', 'new_cases_smoothed_per_million',
#        'total_deaths_per_million', 'new_deaths_per_million',
#        'new_deaths_smoothed_per_million', 'icu_patients',
#        'icu_patients_per_million', 'weekly_hosp_admissions',
#        'weekly_hosp_admissions_per_million', 'total_tests',
#        'total_tests_per_thousand', 'new_tests_smoothed',
#        'new_tests_smoothed_per_thousand', 'positive_rate', 'tests_per_case',
#        'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
#        'new_vaccinations', 'new_vaccinations_smoothed',
#        'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
#        'people_fully_vaccinated_per_hundred',
#        'new_vaccinations_smoothed_per_million', 'stringency_index',
#        'excess_mortality', 'BanOnAllEvents', 'BanOnAllEventsPartial',
#        'ClosDaycare', 'ClosDaycarePartial', 'ClosPrim', 'ClosPrimPartial',
#        'ClosPubAny', 'ClosPubAnyPartial', 'ClosSec', 'ClosSecPartial',
#        'EntertainmentVenues', 'EntertainmentVenuesPartial',
#        'GymsSportsCentres', 'GymsSportsCentresPartial',
#        'HotelsOtherAccommodationPartial', 'IndoorOver100', 'IndoorOver1000',
#        'IndoorOver50', 'MasksMandatoryAllSpacesPartial',
#        'MasksMandatoryClosedSpaces', 'MasksVoluntaryAllSpaces', 'MassGather50',
#        'MassGather50Partial', 'MassGatherAll', 'MassGatherAllPartial',
#        'NonEssentialShops', 'NonEssentialShopsPartial', 'OutdoorOver100',
#        'PlaceOfWorshipPartial', 'PrivateGatheringRestrictions',
#        'QuarantineForInternationalTravellers',
#        'QuarantineForInternationalTravellersPartial', 'RestaurantsCafes',
#        'RestaurantsCafesPartial', 'SocialCirclePartial', 'StayHomeGen',
#        'StayHomeGenPartial', 'StayHomeOrderPartial', 'Teleworking',
#        'TeleworkingPartial', 'tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd',
#        'wpgt', 'pres',],
#                     class_names=(['decreasing case numbers', 'increasing case numbes']), rounded=True, filled=True)
#     os.system("dot -Tpng Covid_tree.dot -o Covid_tree.png")
#     os.system("dot -Tps Covid_tree.dot -o Covid_tree.ps")
#
#     # change to the start working directory
#     os.chdir(path_start)
#     return



def accuracy_plot(Akkuranz, algo):
    # change the working directory
    path_start = os.getcwd()
    pathr = os.path.dirname(os.getcwd()) + '/covidCasePredictions/reports/figures'
    os.chdir(pathr)

    # define figure size
    plt.figure(figsize=(8, 8))

    # add title
    plt.title('{} accuracy versus train/test data size ratio'.format(algo))

    # plot and add labels to plot
    plt.plot(np.linspace(0.2, 0.8, 7), Akkuranz, 'b--*', label='K_fold Accuracy with best parameters')
    plt.legend(loc='lower right')
    plt.xlabel('test/train ratio')
    plt.ylabel('K_fold Accuracy')

    plt_name = "AccuracySize_" + algo + ".png"
    plt.savefig(plt_name)
    # savefig options: dpi = 96pxl, transparent= True, bbox_inches='tight', pad_inches

    # plt.show()

    # change to the start working directory
    os.chdir(path_start)


# Correlation matrix
def correlation(data):
    corr = data.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
       ax.get_xticklabels(),
       rotation=45,
       horizontalalignment='right'
    )




# löschen, ist in Classifier integriert
# def confmat_plot(cm, alg, vot):
#     rcParams['figure.figsize'] = 40, 12
#     # Nam = []
#     # Nam = alg.append(vot)
#
#     # Nam = ["DT","rf","knn","svm",'hard','soft','softWithWeight','softCutoff','softWWCutoff']
#     name = ["dt", "rf", "knn", 'hard', 'soft', 'softWithWeight', 'softCutoff', 'softWWCutoff']
#
#     # change the working directory
#     path_start = os.getcwd()
#     pathr = os.path.dirname(os.getcwd()) + '/covidCasePredictions/reports/figures'
#     os.chdir(pathr)
#
#     # define figure size
#     # axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(12, 8))
#
#     # add title
#     # axarr.set_title(Nam)
#     for i in range(0, len(cm)):
#         # define figure size
#         plt.figure(figsize=(4, 4))
#
#         cm_display = ConfusionMatrixDisplay(cm[i]).plot()
#
#         plt_name = "ConfMat" + name[i] + ".png"
#         plt.savefig(plt_name)
#
#     # change to the start working directory
#     os.chdir(path_start)