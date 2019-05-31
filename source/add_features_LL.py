# This file contains functions to add features.
#
# Lauren Li

import sqlite3
import pandas as pd
import numpy as np
import features as ft
import gettinglabels

DATABASE_FILENAME="../ncdoc_data/data/preprocessed/inmates.db"
# priority 1

def create_labels_indices(database_path=DATABASE_FILENAME):
    create_index = (""" 
    CREATE INDEX IF NOT EXISTS
    idx_start
    ON labels(START_DATE)
    """,
    """
    CREATE INDEX IF NOT EXISTS
    idx_end
    ON labels(END_DATE)
    """)
    ft.build_index(database_path, create_index)

def create_OFNT3CE1_indices(database_path=DATABASE_FILENAME):
    create_index = (""" 
    CREATE INDEX IF NOT EXISTS
    idx_offense_start
    ON OFNT3CE1(DATE_OFFENSE_COMMITTEDBEGIN)
    """,
    """
    CREATE INDEX IF NOT EXISTS
    idx_offense_end
    ON OFNT3CE1(DATE_OFFENSE_COMMITTEDEND)
    """,
    """
    CREATE INDEX IF NOT EXISTS
    idx_OFNT3CE1_id
    ON OFNT3CE1(OFFENDER_NC_DOC_ID_NUMBER)
    """)
    ft.build_index(database_path, create_index)

def add_gender_race_age(database_path=DATABASE_FILENAME):
    create_labels_indices()

    query = (
    """
    WITH inmate_char as (
        SELECT labels.ID, PREFIX, INMATE_GENDER_CODE, INMATE_RACE_CODE, START_DATE, END_DATE, INMATE_BIRTH_DATE,
        CASE WHEN (julianday(START_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 < 0
        THEN NULL
        ELSE (julianday(START_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 END as AGE_AT_START_DATE, 
        CASE WHEN (julianday(END_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 < 0
        THEN NULL
        ELSE (julianday(END_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 END as AGE_AT_END_DATE
        FROM
        labels LEFT JOIN INMT4AA1
        ON INMT4AA1.INMATE_DOC_NUMBER = labels.ID
    ),
    prev as (
        SELECT ID, PREFIX, ROW_NUMBER() OVER (PARTITION BY ID ORDER BY START_DATE) -1 as NUM_PREV_INCARC,
        CASE WHEN ((ROW_NUMBER() OVER (PARTITION BY ID ORDER BY START_DATE) -1) >= 1)
        THEN 1
        ELSE 0 END AS PREV_INCAR_INDIC
        FROM labels
        ORDER BY START_DATE
    )
    SELECT *
    FROM inmate_char LEFT JOIN prev
    ON inmate_char.ID = prev.ID
    AND inmate_char.PREFIX = prev.PREFIX
    """,)
    table_names = ['inmate_char']
    ft.create_ft_table(database_path, table_names, query)

    indexq = (
    """
    CREATE INDEX IF NOT EXISTS
    idx_birth
    ON inmate_char(INMATE_BIRTH_DATE)
    """,)
    ft.build_index(database_path=DATABASE_FILENAME, query = indexq)

def add_ages(database_path=DATABASE_FILENAME):
    create_OFNT3CE1_indices()
    query = ("""
    WITH age_first as(
        SELECT ID, 
        CASE WHEN (julianday(min(START_DATE)) - julianday(min(INMATE_BIRTH_DATE)))/365.0 < 0 
        THEN NULL
        ELSE (julianday(min(START_DATE)) - julianday(min(INMATE_BIRTH_DATE)))/365.0 END as AGE_FIRST_SENTENCE
        FROM inmate_char
        group by ID
    ),
    age_offense as (
        SELECT inmate_char.ID, inmate_char.PREFIX, OFNT3CE1.min_d as OFFENSE_START, OFNT3CE1.max_d as OFFENSE_END,
        CASE WHEN (julianday(OFNT3CE1.min_d) - julianday(inmate_char.INMATE_BIRTH_DATE))/365.0 < 0
        THEN NULL
        ELSE (julianday(OFNT3CE1.min_d) - julianday(inmate_char.INMATE_BIRTH_DATE))/365.0 END as AGE_AT_OFFENSE_START,
        CASE WHEN (julianday(OFNT3CE1.max_d) - julianday(inmate_char.INMATE_BIRTH_DATE))/365.0 < 0
        THEN NULL
        ELSE (julianday(OFNT3CE1.max_d) - julianday(inmate_char.INMATE_BIRTH_DATE))/365.0 END as AGE_AT_OFFENSE_END
        FROM inmate_char natural join 
        (select OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX, min(DATE_OFFENSE_COMMITTEDBEGIN) as min_d, max(DATE_OFFENSE_COMMITTEDEND) as max_d
        from OFNT3CE1
        group by OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX) as OFNT3CE1
    )
    SELECT age_offense.ID, age_offense.PREFIX, age_offense.OFFENSE_START, age_offense.OFFENSE_END, age_offense.AGE_AT_OFFENSE_START,
    age_offense.AGE_AT_OFFENSE_END, age_first.AGE_FIRST_SENTENCE  FROM
    age_offense LEFT JOIN age_first
    ON age_offense.ID = age_first.ID
    """,)
    table_names = ['age_features']
    ft.create_ft_table(database_path, table_names, query)

def add_num_sentences(database_path=DATABASE_FILENAME):
    query = ("""
    WITH sent as (
        SELECT OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX, count(SENTENCE_COMPONENT_NUMBER) as NUM_SENTENCES
        FROM OFNT3CE1 
        GROUP BY OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX
    )
    SELECT labels.ID, labels.PREFIX, labels.START_DATE, labels.END_DATE, sent.NUM_SENTENCES 
    FROM labels LEFT JOIN sent
    ON labels.ID = sent.ID
    AND labels.PREFIX = sent.PREFIX
    """,)
    table_names = ['num_sent']
    ft.create_ft_table(database_path, table_names, query)

    table_names2 = ['totcntavg_sentences_allprior', 'totcntavg_sentences_last5yr']

    query2 = (''' 
        SELECT
        a.ID, a.PREFIX, a.START_DATE, a.END_DATE,
        sum(b.NUM_SENTENCES) as TOTAL_SENT_ALLPRIOR,
        count(*) as NUM_PREV_SENT_ALLPRIOR,
        sum(b.NUM_SENTENCES)/count(*) as AVG_SENT_ALLPRIOR 
        FROM num_sent as a join num_sent as b on a.ID=b.ID
        WHERE b.END_DATE <= a.END_DATE 
        GROUP BY a.ID, a.PREFIX, a.START_DATE, a.END_DATE
        ''',
        '''
        SELECT
        a.ID, a.PREFIX, a.START_DATE, a.END_DATE,
        sum(b.NUM_SENTENCES) as TOTAL_SENT_LAST5YR,
        count(*) as NUM_PREV_SENT_LAST5YR,
        sum(b.NUM_SENTENCES)/count(*) as AVG_SENT_LAST5YR  
        FROM num_sent as a join num_sent as b on a.ID=b.ID 
        WHERE julianday(b.END_DATE) >= (julianday(a.END_DATE) - 1825) 
        AND julianday(b.END_DATE) <= julianday(a.END_DATE)
        GROUP BY a.ID, a.PREFIX, a.START_DATE, a.END_DATE
        '''
        )

    ft.create_ft_table(database_path, table_names2, query2)

# priority 2



# create missing binary variable
# use for: race
# use for: missing (i.e negative) age
def missing_col(df, col):
    '''
    This function creates binary column with missing or not

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
    add_num_sentences()
    #testing impute_age()
    # x = [{'year': 1995, 'penalty': 1, 'age': 25, 'race': 'Black'},
    #     {'year': 1995, 'penalty': 1, 'age': 27, 'race': 'Asian'},
    #     {'year': 1995, 'penalty': 1, 'age': None, 'race': 'White'},
    #     {'year': 1997, 'penalty': 2, 'age': 25, 'race': None},
    #     {'year': 1997, 'penalty': 2, 'age': 30, 'race': None},
    #     {'year': 1997, 'penalty': 2, 'age': None, 'race': 'Indian'}]
    # y = pd.DataFrame(x)
    # z=impute_age(y, 'year', 'penalty', 'age')
    # print(z)

    # a = impute_race(z, 'race')
    # print(a)