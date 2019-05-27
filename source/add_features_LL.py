# This file contains functions to add features.
#
# Lauren Li

import sqlite3

def add_gender_race_age_at_start_end():
    query = """
    SELECT ID, PREFIX, INMATE_GENDER_CODE, INMATE_RACE_CODE, START_DATE, END_DATE, INMATE_BIRTH_DATE,
    (julianday(START_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 as AGE_AT_START_DATE, 
    (julianday(END_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 as AGE_AT_END_DATE
    FROM
    INMT4AA1 JOIN labels
    ON INMT4AA1.INMATE_DOC_NUMBER = labels.ID
    """
    return query

def add_age_at_sentence():
    query = """

    """

# create missing binary variable
# use for: race
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
    df.loc[df[col].isna(), col] = fill_method(df[col])
    return df