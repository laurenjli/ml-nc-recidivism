#!/usr/bin/env python
# coding: utf-8

from pipeline import *
import pandas as pd
import traintestset as tt


#Define constants
DATA_DIR = "../ncdoc_data/data/preprocessed/traintest"
RESULTS_DIR = "results"
RESULTS_FILE = "results.csv"

#MODIFY THESE DICTIONARIES TO HAVE THE PARAMETERS NEEDED FOR CLEANING
CLEAN_PARAMETERS = {'years_range': [""], 'sd_threshold': 3,\
                    'how_fill_nan':'median'} #we might want the how_fill_nan to be a dictionary that changes for each variable.  

VARIABLES = {
             'LABEL' : 'LABEL',
             'TEMPORAL_VAL_VAR' : '',
             'DATE_OUTCOME' : '',
             'IDENTIFICATION_VARS' : [''],
             'FLAG_VARS' : [''],
             'CONTINUOUS_VARS' : [''],
             'CATEGORICAL_VARS' : [''],
             'VARS_TO_DROP' : [''],
            }


#For this progress report I only ran the following small grid:
CUSTOM_GRID = {
'LR': { 'penalty': ['l1','l2'], 'C': [0.01,0.1,1,10], 'random_state': [SEED]},
'KNN' :{'n_neighbors': [3,5,10],'weights': ['uniform'],'algorithm': ['auto']},
'DT': {'criterion': ['gini'], 'max_depth': [1,5,10],'min_samples_split': [10, 20, 50], 'random_state': [SEED]},
'RF':{'n_estimators': [1,10,100], 'max_depth': [5], 'max_features': ['sqrt'],'min_samples_split': [10, 20, 50], 'random_state': [SEED]},
'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1,10,100], 'random_state': [SEED]},
'BA': {'base_estimator': [LogisticRegression()], "n_estimators":[1,10,100], 'random_state': [SEED]},
'SVM' :{'C' :[0.01,0.1,1,10], 'tol':[1e-5], 'random_state': [SEED]}
}
'''

CUSTOM_GRID = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'random_state':[0], 'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10], 'tol':[1e-5]},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
     'BA': {'base_estimator': [LogisticRegression()], "n_estimators":[1]}}
    #'SVM' :{'C' :[0.01,0.1,1,10], 'tol':[1e-5], 'random_state': [SEED]},
'''

EVAL_METRICS_BY_LEVEL = (['accuracy', 'precision', 'recall', 'f1'], [1,2,5,10,20,30,50])
EVAL_METRICS = ['auc']
MODELS = ['LR', 'SVM', 'DT','KNN', 'RF', 'AB', 'BA']
SEED = 0

#def main(dir=DATA_DIR, files=FILE_NAMES, label=LABEL, results_file_name=RESULTS_FILE):
def main(data_dir=DATA_DIR, results_dir=RESULTS_DIR, results_file=RESULTS_FILE, 
         variables=VARIABLES, clean_parameters=CLEAN_PARAMETERS, period=[1997, 2017]):
    
    year = period[0]
    label = variables['LABEL']

    while year <= period[1]:
        
        test_csv = os.path.join(data_dir, "test_{}_test.csv".format(year))
        train_csv = os.path.join(data_dir, "test_{}_train.csv".format(year))
        
        if not os.path.exists(test_csv) and os.path.exists(train_csv):
            tt.full_traintest() 

        df_test = get_csv(dir, test_set)
        df_train = get_csv(dir, train_set)
                            
        #DATA CLEANING HERE

        attributes_lst = ['INCARCERATION_LEN_DAYScat']

        results = classify(df_train, df_test, LABEL, MODELS, EVAL_METRICS, EVAL_METRICS_BY_LEVEL, CUSTOM_GRID, attributes_lst)
        results['year'] = year

        if year == first_year:
            results.to_csv(os.path.join(results_dir, results_file), index=False)
        else:
            with open(os.path.join(results_dir, results_file), 'a') as f:
                results.to_csv(f, header=False, index=False) 


        year += 1

if __name__ == "__main__":
  main()
