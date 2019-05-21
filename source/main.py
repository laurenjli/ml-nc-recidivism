#!/usr/bin/env python
# coding: utf-8

from pipeline import *
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Define constants
DATA_DIR = "../ncdoc_data/data/preprocessed/traintest"
RESULTS_FILE = "results.csv"
LABEL = "LABEL"

#For this progress report I only ran the following small grid:
CUSTOM_GRID = {
'LR': { 'penalty': ['l1','l2'], 'C': [0.1,1]},
'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
'SVM' :{'random_state':[0], 'tol':[1e-5]},
'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
'BA': {'base_estimator': [LogisticRegression()], "n_estimators":[1]}}
'''
CUSTOM_GRID = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'random_state':[0], 'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10], 'tol':[1e-5]},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
     'BA': {'base_estimator': [LogisticRegression()], "n_estimators":[1]}}
    #'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10]},
'''

EVAL_METRICS_BY_LEVEL = (['accuracy', 'precision', 'recall', 'f1'], [1,2,5,10,20,30,50])
EVAL_METRICS = ['auc']
MODELS = ['LR', 'DT', 'SVM','KNN', 'RF', 'AB', 'BA']

#def main(dir=DATA_DIR, files=FILE_NAMES, label=LABEL, results_file_name=RESULTS_FILE):
def main(dir=DATA_DIR, label=LABEL, results_file_name=RESULTS_FILE, first_year=1997, last_year=2018):
    
    year = first_year

    while year < last_year + 1:
        
        test_set = "test_{}_test.csv".format(year)
        train_set = "test_{}_train.csv".format(year)
        
        df_test = get_csv(dir, test_set)
        df_train = get_csv(dir, train_set)

        fill_nan(df_test, ['INCARCERATION_LEN_DAYS', 'LABEL'], how='median')
        fill_nan(df_train, ['INCARCERATION_LEN_DAYS', 'LABEL'], how='median')

#        df_test = remove_outliers(df_test, ['INCARCERATION_LEN_DAYS'], sd_threshold=3)
#        df_train = remove_outliers(df_train, ['INCARCERATION_LEN_DAYS'], sd_threshold=3)

        df_test = discretize_variable(df_test, ["INCARCERATION_LEN_DAYS"])
        df_train = discretize_variable(df_train, ["INCARCERATION_LEN_DAYS"])

        '''
        #df_test = categorical_to_dummy(df_test, ["PREFIX"])
        #df_train = categorical_to_dummy(df_train, ["PREFIX"])
        
        attributes_lst = list(df_test.columns)
        attributes_lst.remove("END_DATE")
        attributes_lst.remove("START_DATE")
        attributes_lst.remove("ID")
        attributes_lst.remove("PREFIX")
        attributes_lst.remove("INCARCERATION_LEN_DAYS")
        attributes_lst.remove("LABEL")
        '''

        attributes_lst = ['INCARCERATION_LEN_DAYScat']

        results = classify(df_train, df_test, LABEL, MODELS, EVAL_METRICS, EVAL_METRICS_BY_LEVEL, CUSTOM_GRID, attributes_lst)
        results['year'] = year

        if year == first_year:
            results.to_csv(results_file_name, index=False)
        else:
            with open(results_file_name, 'a') as f:
                results.to_csv(f, header=False, index=False) 
        
        year += 1

if __name__ == "__main__":
  main()
