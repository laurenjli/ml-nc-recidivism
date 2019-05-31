#!/usr/bin/env python
# coding: utf-8

from pipeline import *
import pandas as pd
import traintestset as tt
import config

#For this progress report I only ran the following small grid:
CUSTOM_GRID = config.define_clfs_params(config.GRIDSIZE)


EVAL_METRICS_BY_LEVEL = (['accuracy', 'precision', 'recall', 'f1'], [1,2,5,10,20,30,50])
EVAL_METRICS = ['auc']
MODELS = ['LR', 'SVM', 'DT','KNN', 'RF', 'AB', 'BA']
SEED = 0

#def main(dir=DATA_DIR, files=FILE_NAMES, label=LABEL, results_file_name=RESULTS_FILE):
def main(data_dir=DATA_DIR, results_dir=RESULTS_DIR, results_file=RESULTS_FILE, 
         variables=VARIABLES, period=[1997, 2017]):
    
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

        attributes_lst = [x for x in df_train.columns if x not in variables['VARS_TO_EXCLUDE']]
        for attr in attributes_lst:
            if attr not in df_train.columsn:
                df_test.loc[:,c] = 0

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
