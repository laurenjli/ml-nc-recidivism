'''
Code to combine loading database, getting labels and features
and temporal validation

Before using this file, you need the downloaded CSV data for NC recidivism from https://github.com/jtwalsh0/ncdoc_data
The data we are using for this analysis was downloaded as of 26 April 2019 
'''

import gettinglabels
from create_sqldb import create_db

import sqlite3


# Edit the following directory based on your data locations
CSVFOLDER="../ncdoc_data/data/preprocessed/"
DATABASE_FILENAME=CSVFOLDER + "inmates.db"


def setup():
    '''
    Load datasets into inmates.db and generate new table with labels
    '''
    create_db(CSVFOLDER)


## EDITED THIS FUNCTION TO GENERATE NEW FEATURES
def add_features(database_path=DATABASE_FILENAME):
    '''
    Creating a new table with features and labels
    '''    
    feature_query = (
        """
        WITH incarceration_len as(
                SELECT ID, PREFIX, START_DATE, END_DATE, 
                julianday(END_DATE)-julianday(START_DATE) as INCARCERATION_LEN_DAYS
                FROM labels
            ),
            totcntavg_incarceration_allprior as (
                SELECT a.ID, a.PREFIX, a.START_DATE, a.END_DATE, 
                sum(b.INCARCERATION_LEN_DAYS) as TOTAL_INCARCERATION_ALLPRIOR,
                count(*) as NUM_PREV_INCARCERATION_ALLPRIOR,
                sum(b.INCARCERATION_LEN_DAYS)/count(*) as AVG_INCARCERATION_ALLPRIOR 
                FROM incarceration_len as a join incarceration_len as b on a.ID=b.ID
                WHERE b.END_DATE <= a.END_DATE 
                GROUP BY a.ID, a.PREFIX, a.START_DATE, a.END_DATE
            ),
            totcntavg_incarceration_last5yr as (
                SELECT a.ID, a.PREFIX, a.START_DATE, a.END_DATE, 
                sum(b.INCARCERATION_LEN_DAYS) as TOTAL_INCARCERATION_LAST5YR,
                count(*) as NUM_PREV_INCARCERATION_LAST5YR,
                sum(b.INCARCERATION_LEN_DAYS)/count(*) as AVG_INCARCERATION_LAST5YR  
                FROM incarceration_len as a join incarceration_len as b on a.ID=b.ID 
                WHERE julianday(b.END_DATE) >= (julianday(a.END_DATE) - 1825)
                GROUP BY a.ID, a.PREFIX, a.START_DATE, a.END_DATE
            ),
            relative_incarceration_len as (
                SELECT ID, PREFIX, START_DATE, END_DATE, 
                INCARCERATION_LEN_DAYS - (SELECT
                sum(b.INCARCERATION_LEN_DAYS)/count(*) as mean
                FROM incarceration_len as a, incarceration_len as b
                WHERE b.END_DATE <= a.END_DATE) as RELATIVE_INCARCERATION_LEN_DAYS
                from incarceration_len
            )

        SELECT * 
        FROM labels 
        natural join incarceration_len 
        natural join totcntavg_incarceration_allprior 
        natural join totcntavg_incarceration_last5yr
        natural join relative_incarceration_len
        """)

    gettinglabels.query_db(feature_query, args=None, database_path=DATABASE_FILENAME, table_name='data', new_table=True)
    print('feature created')


def temporal_validation(csv_name, train_start_year, test_start_year, time_period=365.0):
    '''
    Splitting up train and test sets based on the year specified
    e.g. train 2015 using 2016 to get labels (defined by time period),
    test on 2017

    To implement: 
        Train data - from train start year to (test start year - timeperiod)
        filter data for end year to be in test set
    '''

    train_query = """
    SELECT * FROM data
    WHERE END_DATE >= ?
    AND julianday(END_DATE) < julianday(?) - ?
    """
    train_args = (str(train_start_year)+'-01-01', str(test_start_year)+'-01-01', time_period)

    gettinglabels.query_db(train_query, train_args, DATABASE_FILENAME, 
                           table_name='traindata', new_table=True, 
                           csv_filename=CSVFOLDER + 'traintest/' + csv_name + '_train.csv')
    print("created {} train set".format(csv_name))

    test_query = """
    SELECT * FROM data
    WHERE END_DATE LIKE ?
    """
    test_args = (str(test_start_year)+'%',)
        
    gettinglabels.query_db(test_query, test_args, DATABASE_FILENAME, 
                           table_name='testdata', new_table=True, 
                           csv_filename=CSVFOLDER + 'traintest/' + csv_name+ '_test.csv')
    print("created {} test set".format(csv_name))


def get_train_test_splits(train_start_year=1995, test_start_year=1997, time_period=365.0):
    '''
    Gets the train data and test data for individual train-test periods
    1) traindata table and test data table in db
    2) write out as csv files in data/preprocessed/traintest/ folder
    '''
    gettinglabels.create_labels(DATABASE_FILENAME, time_period=time_period, default_max = 10000.0, table_name = 'labels')
    add_features()
    temporal_validation('test_'+ str(test_year), train_start_year=train_start_year, test_start_year=test_start_year, time_period=time_period)


if __name__ == '__main__':
    time_period = 365.0

#    setup()  # To load database of data tables
#    gettinglabels.create_labels(DATABASE_FILENAME, time_period=time_period, default_max = 10000.0, table_name = 'labels')  # Get labels
    add_features() # Create new table data for features and data

#    test_year=1997
#    while test_year <= 2018:
#        temporal_validation('test_'+ str(test_year), train_start_year=1995, test_start_year=test_year, time_period=time_period)
#        test_year += 1


