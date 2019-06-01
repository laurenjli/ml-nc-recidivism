#!/usr/bin/env python
# coding: utf-8

import pipeline as pp
import pandas as pd
import traintestset as tt
import datetime as dt
import config
import os



#def main(dir=DATA_DIR, files=FILE_NAMES, label=LABEL, results_file_name=RESULTS_FILE):
def main(data_dir=config.DATA_DIR, results_dir=config.RESULTS_DIR, results_file=config.RESULTS_FILE, 
         variables=config.VARIABLES, models=config.MODELS, eval_metrics=config.EVAL_METRICS,
         eval_metrics_by_level=config.EVAL_METRICS_BY_LEVEL, grid=config.define_clfs_params(config.GRIDSIZE), 
         period=[1997, 2017]):
    
    
    year = period[0]
    label = config.VARIABLES['LABEL']

    while year <= period[1]:
        
        test_csv = os.path.join(data_dir, "test_{}_test.csv".format(year))
        train_csv = os.path.join(data_dir, "test_{}_train.csv".format(year))
        
        if not os.path.exists(test_csv) or not os.path.exists(train_csv):
            print('Creating training and test sets')
            tt.full_traintest() 

        df_test = pp.get_csv(test_csv)
        df_train = pp.get_csv(train_csv)

        # PRE PROCESS DATA

        # changing data type to date
        for df in [df_test,  df_train]:
            pp.to_date(df, variables['DATES'])
        # create year of sentence column for future imputation
            df['SENTENCE_YEAR'] = df['START_DATE'].dt.year

            ## outliers

            ## create missing indicators
            for fill in variables['INDICATOR']:
                attributes = variables['INDICATOR'][fill]
                for attr in attributes:
                    if attr == 'INCARCERATION_LEN_DAYS':
                        df[attr] = df[attr].where(df[attr]>=0)   # replace neg values with nulls
                    df = pp.create_indicator(df, attr, fill)
            
            ## missing imputation
            for attribute in variables['MISSING']['AGE']:
                year_col = 'SENTENCE_YEAR'
                pen_col = 'INMATE_RACE_CODE'
                df = pp.impute_with_2cols(df, year_col, pen_col, attribute)
            
            for attribute in variables['MISSING']['MISSING_CAT']:
                df = pp.impute_missing(df, attribute)

            for attribute in variables['MISSING']['IMPUTE_MEAN']:
                df = pp.na_fill_col(df, attribute)

            ## discretization

            ## dummy
            for attribute in variables['CATEGORICAL_VARS']:
                if attribute == 'MINMAXTERM':
                    r = {'MAX.TERM:,MIN.TERM:': 'MIN.TERM:,MAX.TERM:'}
                    df['MINMAXTERM'] = df['MINMAXTERM'].replace(r)

            pp.categorical_to_dummy(df, variables['CATEGORICAL_VARS'])

        #scaling continuous variable
        scaler = MinMaxScaler()
        for attribute in variables['CONTINUOUS_VARS_MINMAX']:
            data_for_fitting = df_train[attribute].values.reshape(-1,1)
            s = scaler.fit(data_for_fitting)
            df_train[attribute] = scaler.transform(df_train[attribute].values.reshape(-1, 1))
            df_test[attribute] = scaler.transform(df_test[attribute].values.reshape(-1, 1))

        # define list of features
        attributes_lst = [x for x in df_train.columns if x not in variables['VARS_TO_EXCLUDE']]
        for attr in attributes_lst:
            if attr not in df_train.columsn:
                df_test.loc[:,c] = 0

        # run models
        results = classify(df_train, df_test, label, models, eval_metrics, eval_metrics_by_level, grid, attributes_lst)
        results[config.TRAIN_TEST_COL] = year

        # save results
        if year == first_year:
            results.to_csv(os.path.join(results_dir, results_file), index=False)
        else:
            with open(os.path.join(results_dir, results_file), 'a') as f:
                results.to_csv(f, header=False, index=False) 

        year += 1

if __name__ == "__main__":
  main()
