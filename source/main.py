#!/usr/bin/env python
# coding: utf-8

import pipeline as pp
import pandas as pd
import traintestset as tt
import config



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
        
        if not os.path.exists(test_csv) and os.path.exists(train_csv):
            tt.full_traintest() 

        df_test = get_csv(dir, test_set)
        df_train = get_csv(dir, train_set)

        for df in [df_test,  df_train]:
            # changing data type
            pp.to_date(df, VARIABLES['DATES'])

            ## outliers

            ## create indicators
            for fill in VARIABLES['INDICATOR']:
                attributes = VARIABLES['INDICATOR'][fill]
                for attr in attribute:
                    if attr == 'INCARCERATION_LEN_DAYS':
                        df[attr] = df[attr].where(df[attr]>=0)   # replace neg values with nulls
                    df = pp.create_indicator(df, attr, fill)
            
            ## missing imputation
            for attribute in VARIABLES['MISSING']['AGE']:
                df = pp.impute_with_2cols(df, year_col, pen_col, attribute)
            
            for attribute in VARIABLES['MISSING']['MISSING_CAT']:
                df = pp.impute_missing(df, attribute)

            for attribute in VARIABLES['MISSING']['IMPUTE_MEAN']:
                df = pp.na_fill_col(df, attribute)

            ## discretization

            ## dummy
            for attribute in VARIABLES['CATEGORICAL_VARS']:
                if attribute == 'MINMAXTERM':
                    r = {'MAX.TERM:,MIN.TERM:': 'MIN.TERM:,MAX.TERM:'}
                    df['MINMAXTERM'] = df['MINMAXTERM'].replace(r)

            pp.categorical_to_dummy(df, VARIABLES['CATEGORICAL_VARS'])

        #scaling continuous variable
        scaler = MinMaxScaler()
        for attribute in VARIABLES['CONTINUOUS_VARS_MINMAX']:
            data_for_fitting = df_train[attribute].values.reshape(-1,1)
            s = scaler.fit(data_for_fitting)
            df_train[attribute] = scaler.transform(df_train[attribute].values.reshape(-1, 1))
            df_test[attribute] = scaler.transform(df_test[attribute].values.reshape(-1, 1))


        attributes_lst = [x for x in df_train.columns if x not in variables['VARS_TO_EXCLUDE']]
        for attr in attributes_lst:
            if attr not in df_train.columsn:
                df_test.loc[:,c] = 0

        results = classify(df_train, df_test, label, models, eval_metrics, eval_metrics_by_level, grid, attributes_lst)
        results[config.TRAIN_TEST_COL] = year

        if year == first_year:
            results.to_csv(os.path.join(results_dir, results_file), index=False)
        else:
            with open(os.path.join(results_dir, results_file), 'a') as f:
                results.to_csv(f, header=False, index=False) 

        year += 1

if __name__ == "__main__":
  main()
