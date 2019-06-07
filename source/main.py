#!/usr/bin/env python
# coding: utf-8

import pipeline as pp
import pandas as pd
import traintestset as tt
import datetime as dt
import config
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import features as ft
import warnings
warnings.filterwarnings('ignore')

def preprocess(df, variables=config.VARIABLES):
    '''
    This function does the data preprocessing for each test, train set.

    df: dataframe
    variables: from config file

    returns df with preprocessed data 
    '''
    # turn date columns to dates
    pp.to_date(df, variables['DATES'])
    # create year of sentence column for future imputation
    df['SENTENCE_YEAR'] = df['START_DATE'].dt.year

    ## create missing indicators
    for fill in variables['INDICATOR']:
        attributes = variables['INDICATOR'][fill]
        for attr in attributes:
            df = pp.create_indicator(df, attr, fill)
    
    ## missing imputation
    for attribute in variables['MISSING']['AGE']:
        year_col = 'SENTENCE_YEAR'
        pen_col = 'PRIMARY_OFFENSE_CODE'
        len_col = 'INCARCERATION_LEN_DAYS'
        df = pp.impute_age(df, year_col, pen_col, len_col, 'AGE_AT_START_DATE', 'AGE_AT_END_DATE', 'AGE_FIRST_SENTENCE')

    
    for attribute in variables['MISSING']['MISSING_CAT']:
        df = pp.impute_missing(df, attribute)

    # for attribute in variables['MISSING']['INCARCERATION']:
    #     offense_col = 'PRIMARY_OFFENSE_CODE'
    #     pen_col = 'SENTENCING_PENALTY_CLASS_CODE'
    #     df = pp.impute_with_2cols(df, offense_col, pen_col, attribute)

    for attribute in variables['MISSING']['IMPUTE_ZERO']:
        df = pp.impute_missing(df, attribute, 0)

    ## dummy
    df = pp.categorical_to_dummy_with_groupconcat(df,variables['SPECIAL_DUMMY'])

    for attribute in variables['CATEGORICAL_VARS']:
        if attribute == 'MINMAXTERM':
            r = {'MAX.TERM:,MIN.TERM:': 'MIN.TERM:,MAX.TERM:'}
            df['MINMAXTERM'] = df['MINMAXTERM'].replace(r)
            
    return pp.categorical_to_dummy(df, variables['CATEGORICAL_VARS'])

#def main(dir=DATA_DIR, files=FILE_NAMES, label=LABEL, results_file_name=RESULTS_FILE):
def main(gender = config.GENDER, data_dir=config.DATA_DIR, results_dir=config.RESULTS_DIR, results_file=config.RESULTS_FILE, graphs_dir = config.GRAPH_FOLDER,
         variables=config.VARIABLES, models=config.MODELS, eval_metrics=config.EVAL_METRICS,
         eval_metrics_by_level=config.EVAL_METRICS_BY_LEVEL, grid=config.define_clfs_params(config.GRIDSIZE), 
         period=config.YEARS, plot_pr = config.PLOT_PR, compute_bias = config.BIAS, save_pred = config.SAVE_PRED):

    # check if necessary data and results directories exist
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if not os.path.exists(graphs_dir):
        os.mkdir(graphs_dir)

    # initialize variables
    first_year = period[0]
    year = period[0]
    label = config.VARIABLES['LABEL']
    bias_lst = variables['BIAS']
    bias_dict = variables['BIAS_METRICS']

    while year <= period[1]:
        print('Running year: {}'.format(year))

        # check if training/test data exists, create it if not
        test_csv = os.path.join(data_dir, "test_{}_test.csv".format(year))
        train_csv = os.path.join(data_dir, "test_{}_train.csv".format(year))
        
        if not os.path.exists(test_csv) or not os.path.exists(train_csv):
            print('Creating training and test sets')
            tt.full_traintest() 

        # read in training and test data 
        df_test = pp.get_csv(test_csv)
        df_train = pp.get_csv(train_csv)

        # filter gender
        if gender == 'MALE_':
            print('Gender filter: male (and missing)')
            df_test = df_test[df_test['INMATE_GENDER_CODE'] != 'FEMALE']
            df_train = df_train[df_train['INMATE_GENDER_CODE'] != 'FEMALE']
        elif gender == 'FEMALE_':
            print('Gender filter: female (and missing)')
            df_test = df_test[df_test['INMATE_GENDER_CODE'] != 'MALE']
            df_train = df_train[df_train['INMATE_GENDER_CODE'] != 'MALE']

        # Pre-process data 
        print('Pre-processing data')
        df_test = preprocess(df_test)
        df_train = preprocess(df_train)

        #scaling continuous variable
        print('Scaling data')
        scaler = MinMaxScaler()
        for attribute in variables['CONTINUOUS_VARS_MINMAX']:
            data_for_fitting = df_train[attribute].values.reshape(-1,1)
            s = scaler.fit(data_for_fitting)
            df_train[attribute] = scaler.transform(df_train[attribute].values.reshape(-1, 1))
            df_test[attribute] = scaler.transform(df_test[attribute].values.reshape(-1, 1))

        # define list of features
        attributes_lst = [x for x in df_train.columns if x not in variables['VARS_TO_EXCLUDE']]
        #bias_lst = variables['BIAS']
        for attr in attributes_lst:
            if attr not in df_test.columns:
                df_test.loc[:,attr] = 0
        print('Training set has {} features'.format(len(attributes_lst)))
        #print(df_train['INMATE_GENDER_CODE'].unique())
        #print(df_test['INMATE_GENDER_CODE'].unique())

        # run models
        results = pp.classify(df_train, df_test, label, models, eval_metrics, eval_metrics_by_level, grid, attributes_lst, 
            bias_lst, bias_dict, year, results_dir, results_file, plot_pr, compute_bias, save_pred)
        # # add year
        # results[config.TRAIN_TEST_COL] = year
        # # add baseline for test set
        # results['baseline'] = sum(df_test[label])/len(df_test[label])

        # # save results
        # if year == first_year:
        #     results.to_csv(os.path.join(results_dir, results_file), index=False)
        # else:
        #     with open(os.path.join(results_dir, results_file), 'a') as f:
        #         results.to_csv(f, header=False, index=False) 

        year += 1

if __name__ == "__main__":
  main()
