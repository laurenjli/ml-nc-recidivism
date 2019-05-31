import pandas as pd
import gettinglabels
import sqlite3

DATABASE_FILENAME="../ncdoc_data/data/preprocessed/inmates.db"


## FEATURE GENERATION ##

def add_features(database_path, table_names, insert_query_list):
    '''
    Generic code to add a new table into DB for the feature
    '''
    # create new tables for new features
    for i, query in enumerate(insert_query_list):
        name = table_names[i]
        gettinglabels.query_db(query, args=None, database_path=database_path, table_name=name, new_table=True)
        print("table {} created".format(name))

