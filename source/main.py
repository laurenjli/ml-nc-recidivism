#!/usr/bin/env python
# coding: utf-8

from pipeline import *
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Define constants
DATA_DIR = "../ncdoc_data/data/preprocessed/traintest"
FILE_NAMES = [("test_1997_test.csv", "test_1997_train.csv"),\
              ("test_1998_test.csv", "test_1998_train.csv"),\
              ("test_1999_test.csv", "test_1999_train.csv"),\
              ("test_2000_test.csv", "test_2000_train.csv"),\
              ("test_2001_test.csv", "test_2001_train.csv"),\
              ("test_2002_test.csv", "test_2002_train.csv"),\
              ("test_2003_test.csv", "test_2003_train.csv"),\
              ("test_2004_test.csv", "test_2004_train.csv"),\
              ("test_2005_test.csv", "test_2005_train.csv"),\
              ("test_2006_test.csv", "test_2006_train.csv"),\
              ("test_2007_test.csv", "test_2007_train.csv"),\
              ("test_2008_test.csv", "test_2008_train.csv"),\
              ("test_2009_test.csv", "test_2009_train.csv"),\
              ("test_2010_test.csv", "test_2010_train.csv"),\
              ("test_2011_test.csv", "test_2011_train.csv"),\
              ("test_2012_test.csv", "test_2012_train.csv"),\
              ("test_2013_test.csv", "test_2013_train.csv"),\
              ("test_2014_test.csv", "test_2014_train.csv"),\
              ("test_2015_test.csv", "test_2015_train.csv"),\
              ("test_2016_test.csv", "test_2016_train.csv"),\
              ("test_2017_test.csv", "test_2017_train.csv")]

RESULTS_FILE = "results"
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
def main(dir=DATA_DIR, label=LABEL, results_file_name=RESULTS_FILE):
    
    year=1997

    while year < 2018:
        
        test_set = "test_{}_test.csv".format(year)
        train_set = "test_{}_train.csv".format(year)
        
        df_test = get_csv(dir, test_set)
        df_train = get_csv(dir, train_set)

        #fill_nan(df_test, ['INCARCERATION_LEN_DAYS', 'PREFIX'], how='mean')
        #fill_nan(df_train, ['INCARCERATION_LEN_DAYS', 'PREFIX'], how='mean')

        df_test = remove_outliers(df_test, ['INCARCERATION_LEN_DAYS', 'PREFIX'], sd_threshold=3)
        df_train = remove_outliers(df_train, ['INCARCERATION_LEN_DAYS', 'PREFIX'], sd_threshold=3)

        df_test = categorical_to_dummy(df_test, ["INCARCERATION_LEN_DAYS"])
        df_train = categorical_to_dummy(df_train, ["INCARCERATION_LEN_DAYS"])

        df_test = discretize_variable(df_test, ["PREFIX"])
        df_train = discretize_variable(df_train, ["PREFIX"])

        attributes_lst = list(df.columns)
        attributes_lst.remove("END_DATE")
        attributes_lst.remove("START_DATE")
        attributes_lst.remove("LABEL")

        results = classify(df_train, df_test, LABEL, N_SAMPLES, DATE_COL, TRAIN_END, TRAIN_DAYS, MODELS, EVAL_METRICS,\
                           EVAL_METRICS_BY_LEVEL, CUSTOM_GRID, attributes_lst)

        results_file_name_year = '{}_{}{}'.format(results_file_name, year, '.csv')
        results.to_csv(results_file_name_year, index=False)
        year += 1

if __name__ == "__main__":
  main()
