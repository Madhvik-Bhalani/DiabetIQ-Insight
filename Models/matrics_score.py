from k_nearest_neighbors import *
from decision_tree_classification import dicision_tree_metrics
from kernel_svm import kernam_svm_metrics
from logistic_regression import logistic_regression_metrics
from naive_bayes import naive_bayes_metrics
from random_forest_classification import random_forest_metrics
from support_vector_machine import svm_metrics
from xg_boost import xgboost_metrics


def final_metrics_score():
    final_score = []
    k_neighbors_data = k_neighbors_metrics()
    dicision_tree_data = dicision_tree_metrics()
    kernal_svm_data = kernam_svm_metrics()
    logistic_regression_data = logistic_regression_metrics()
    naive_bayes_data = naive_bayes_metrics()
    random_forest_data = random_forest_metrics()
    svm_data = svm_metrics()
    xgboost_data = xgboost_metrics()

    final_score.append(k_neighbors_data)
    final_score.append(dicision_tree_data)
    final_score.append(kernal_svm_data)
    final_score.append(logistic_regression_data)
    final_score.append(naive_bayes_data)
    final_score.append(random_forest_data)
    final_score.append(svm_data)
    final_score.append(xgboost_data)
    
    return final_score


