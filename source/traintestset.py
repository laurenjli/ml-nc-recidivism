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


def add_features(database_path=DATABASE_FILENAME):
    '''
    Creating a new table with features and labels
    '''    
    # First feature - how long they have been incarcerated in days
    feature_query = (
        """
        WITH incarceration_len as(
            SELECT ID, PREFIX, julianday(END_DATE)-julianday(START_DATE) as INCARCERATION_LEN_DAYS
            FROM labels
        )
        SELECT * 
        FROM labels natural join incarceration_len
        """)

    gettinglabels.query_db(feature_query, args=None, database_path=DATABASE_FILENAME, table_name='data', new_table=True)
    print('feature created')


def temporal_validation(train_start_year=1995, test_start_year=1997, time_period=365.0):
    '''
    Splitting up train and test sets based on the year specified
    e.g. train 2015 using 2016 to get labels (defined by time period),
    test on 2017

    To implement: 
    filter data from start year to end year -365 days to be in train set
    filter data for end year to be in test set
    '''
    # Generating the Labels - labels table in db
    gettinglabels.create_labels(DATABASE_FILENAME, time_period = time_period, default_max = 10000.0, table_name = 'labels')

    # Adding on Features to the database - data table in db
    add_features()

    # Splitting Train and Test sets
    # conn = sqlite3.connect(DATABASE_FILENAME)
    # gettinglabels.query_db(train_query, args, DATABASE_FILENAME, table_name='traindata', new_table=True)
    # gettinglabels.query_db(test_query, args, DATABASE_FILENAME, table_name='testdata', new_table=True)
    
    # command = '''
    # select * from labels
    # '''

    # df = pd.read_sql_query(command, conn)

    # https://www.dataquest.io/blog/python-pandas-databases/

if __name__ == '__main__':
#    setup() # To load database of data tables
    temporal_validation(train_start_year=1995, test_start_year=1997, time_period=365.0)



