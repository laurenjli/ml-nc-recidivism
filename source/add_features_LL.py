# This file contains functions to add features.
#
# Lauren Li

import sqlite3
import pandas as pd
import numpy as np

def add_gender_race_age_at_start_end():
    query = """
    SELECT ID, PREFIX, INMATE_GENDER_CODE, INMATE_RACE_CODE, START_DATE, END_DATE, INMATE_BIRTH_DATE,
    CASE WHEN (julianday(START_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 < 0
    THEN NULL
    ELSE (julianday(START_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 END as AGE_AT_START_DATE, 
    CASE WHEN (julianday(END_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 < 0
    THEN NULL
    ELSE (julianday(END_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 END as AGE_AT_END_DATE
    FROM
    INMT4AA1 JOIN labels
    ON INMT4AA1.INMATE_DOC_NUMBER = labels.ID
    """
    return query

def add_num_sentences():
    query = """
    SELECT OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX, count(SENTENCE_COMPONENT_NUMBER) as NUM_SENTENCES
    FROM OFNT3CE1 
    GROUP BY OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX
    """
    return query

# create missing binary variable
# use for: race
# use for: missing (i.e negative) age
def missing_col(df, col):
    '''
    This function fills NA values in a df column by given method

    df: dataframe
    fill_method: function specifying how to fill the NA values
    col: column name

    return: None
    '''
    missingcol = col + '_missing'
    df[missingcol] = [1 if x else 0 for x in y[col].isna()]
    return df

# fill na values
# use for: race (fill missing race with Unknown by using fill_method = 'Unknown')
def na_fill_col(df, col, fill_method = np.mean):
    '''
    This function fills NA values in a df column by given method

    df: dataframe
    fill_method: function specifying how to fill the NA values
    col: column name

    return: None
    '''
    cp = df.copy()
    cp.loc[cp[col].isna(), col] = fill_method(cp[col])
    df[col] = cp[col]
    return df

def impute_race(df, racecol = 'INMATE_RACE_CODE', fill_method = 'Unknown'):
    '''
    This function creates a missing binary column and imputes race

    df: dataframe
    fill_method: impute value

    returns dataframe with imputed data
    '''
    # create binary missing column
    df = missing_col(df, racecol)
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
    df = missing_col(df, target_col)

    # copy dataframe to impute values then reinsert into df
    cp = df.copy()
    cp[target_col]= cp.apply(lambda row: find_mean(row[year_col], row[pen_col]) if pd.isnull(row[target_col]) else row[target_col], axis=1)
    cp.loc[cp[target_col].isna(), target_col] = find_mean(cp[year_col], cp[pen_col])
    df[target_col] = cp[target_col]
    return df

if __name__ == '__main__':
    #testing impute_age()
    x = [{'year': 1995, 'penalty': 1, 'age': 25, 'race': 'Black'},
        {'year': 1995, 'penalty': 1, 'age': 27, 'race': 'Asian'},
        {'year': 1995, 'penalty': 1, 'age': None, 'race': 'White'},
        {'year': 1997, 'penalty': 2, 'age': 25, 'race': None},
        {'year': 1997, 'penalty': 2, 'age': 30, 'race': None},
        {'year': 1997, 'penalty': 2, 'age': None, 'race': 'Indian'}]
    y = pd.DataFrame(x)
    z=impute_age(y, 'year', 'penalty', 'age')
    print(z)

    a = impute_race(z, 'race')
    print(a)