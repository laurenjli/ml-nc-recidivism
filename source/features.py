import pandas as pd
import gettinglabels
import sqlite3
import pipeline as pp

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

def build_index(database_path, query):
    '''
    Generic code to build index
    '''
    con = sqlite3.connect(database_path)
    cur = con.cursor()
    for q in query:
        cur.execute(q)
        con.commit()
        print('index done')
    con.close()

## FEATURE CLEANING ##

def impute_race(df, racecol = 'INMATE_RACE_CODE', fill_method = 'Unknown'):
    '''
    This function creates a missing binary column and imputes race

    df: dataframe
    fill_method: impute value

    returns dataframe with imputed data
    '''
    # create binary missing column
    df = pp.missing_col(df, racecol)
    # copy dataframe to impute values then reinsert into df
    cp = df.copy()
    cp.loc[cp[racecol].isna(), racecol] = fill_method
    df[racecol] = cp[racecol]
    return df

def impute_age(df, year_col, pen_col, target_col):
    '''
    This function creates a missing binary column and imputes age using the helper column values given year of crime and sentencing penalty class code

    df: dataframe
    year_col: colname with date of offense
    pen_col: colname with penalty class code info
    target_col: column to impute

    returns dataframe with imputed data
    '''
    # find mean age given year and penalty class
    def find_mean(year, pen_class):
        #print(year)
        #print(pen_class)
        tmp = df[(df[year_col] == year) & (df[pen_col] == pen_class)]
        print(tmp)
        return np.mean(tmp[target_col])
    
    # create binary missing column 
    df = pp.missing_col(df, target_col)

    # copy dataframe to impute values then reinsert into df
    cp = df.copy()
    cp[target_col]= cp.apply(lambda row: find_mean(row[year_col], row[pen_col]) if pd.isnull(row[target_col]) else row[target_col], axis=1)
    cp.loc[cp[target_col].isna(), target_col] = find_mean(cp[year_col], cp[pen_col])
    df[target_col] = cp[target_col]
    return df