'''
Code to combine loading database, getting labels and features
and temporal validation

Before using this file, you need the downloaded CSV data for NC recidivism from https://github.com/jtwalsh0/ncdoc_data
The data we are using for this analysis was downloaded as of 26 April 2019 
'''
import os
import gettinglabels
from create_sqldb import create_db
import features as ft
import config

import sqlite3


# Edit the following directory based on your data locations

def setup():
    '''
    Load datasets into inmates.db and generate new table with labels
    '''
    print('creating DB')
    create_db(config.CSVFOLDER)


def temporal_validation(csv_name, train_start_year, test_start_year, time_period=365.0):
    '''
    Splitting up train and test sets based on the year specified
    e.g. train 2015 using 2016 to get labels (defined by time period),
    test on 2017

    To implement: 
        Train data - from train start year to (test start year - timeperiod)
        filter data for end year to be in test set
    '''
    print('getting train test split {}'.format(test_start_year))

    train_query = """
    SELECT * FROM data
    WHERE END_DATE >= ?
    AND julianday(END_DATE) < julianday(?) - ?
    """
    train_args = (str(train_start_year)+'-01-01', str(test_start_year)+'-01-01', time_period)

    gettinglabels.query_db(train_query, train_args, config.DATABASE_FILENAME, 
                           table_name='traindata', new_table=True, 
                           csv_filename=config.DATA_DIR + '/'+csv_name + '_train.csv')
    print("created {} train set".format(csv_name))

    test_query = """
    SELECT * FROM data
    WHERE END_DATE LIKE ?
    """
    test_args = (str(test_start_year)+'%',)
        
    gettinglabels.query_db(test_query, test_args, config.DATABASE_FILENAME, 
                           table_name='testdata', new_table=True, 
                           csv_filename=config.DATA_DIR + '/'+csv_name+ '_test.csv')
    print("created {} test set".format(csv_name))


def get_train_test_splits(train_start_year=1995, test_start_year=1997, time_period=365.0):
    '''
    Gets the train data and test data for individual train-test periods
    1) traindata table and test data table in db
    2) write out as csv files in data/preprocessed/traintest/ folder
    '''
    gettinglabels.create_labels(config.DATABASE_FILENAME, time_period=time_period, default_max = 10000.0, table_name = 'labels')
    add_features()
    temporal_validation('test_'+ str(test_year), train_start_year=train_start_year, test_start_year=test_start_year, time_period=time_period)


def full_traintest(time_period=365.0):
    # check if necessary directories exist
    if not os.path.exists(config.CSVFOLDER):
        os.mkdir(config.CSVFOLDER)

    # check if data exists
    data = ['OFNT3CE1','INMT4AA1', 'INMT4BB1', 'INMT9CF1', \
            'OFNT1BA1', 'OFNT3BB1', 'OFNT3DE1', 'INMT4CA1']
    
    for table in data:
        filedir = os.path.join(config.CSVFOLDER, "{}.csv".format(table))
        if not os.path.exists(filedir):
            print('Have you downloaded {}?'.format(table))
            print('Check README for data required.')
            return

    # check if full database exists
    if not os.path.exists(config.DATABASE_FILENAME):
        setup()  # To load database of data tables

    # Get labels
    gettinglabels.create_labels(config.DATABASE_FILENAME, time_period=time_period, default_max = 10000.0, table_name = 'labels')
    # Create new tables for features and data
    ft.add_all_features()

    # Create train test sets
    if not os.path.exists(config.DATA_DIR):
        os.mkdir(config.DATA_DIR)
    test_year=1997
    while test_year <= 2017:
        temporal_validation('test_'+ str(test_year), train_start_year=1995, test_start_year=test_year, time_period=time_period)
        test_year += 1

if __name__ == '__main__':
    full_traintest()





