'''
Code for creating the following features: 
- Total/Average incarceration len ever or in the last 5 years
- Count of prev incarceration ever or in the last 5 years
- County of conviction
- Min Max Term

Rachel Ker
'''
import pandas as pd
import gettinglabels
import sqlite3
import pipeline as pp
import features as ft 

DATABASE_FILENAME="../ncdoc_data/data/preprocessed/inmates.db"


## FEATURE GENERATION ##


def add_incarceration_lens(database_path=DATABASE_FILENAME):
    '''
    Features:
        Total/Average incarceration len ever or in the last 5 years
        Count of prev incarceration ever or in the last 5 years
    '''
    table_names = ['incarceration_len']
    query = ('''
        WITH incarceration_len as(
                SELECT ID, PREFIX, START_DATE, END_DATE, 
                julianday(END_DATE)-julianday(START_DATE) as INCARCERATION_LEN_DAYS
                FROM labels)
        SELECT * 
        FROM labels 
        natural join incarceration_len 
        ''',)
    ft.add_features(database_path, table_names, query)
    create_index_incarcerationlen()

    table_names = ['totcntavg_incarceration_allprior', 'totcntavg_incarceration_last5yr']

    query = (''' 
        SELECT
        a.ID, a.PREFIX, a.START_DATE, a.END_DATE,
        sum(b.INCARCERATION_LEN_DAYS) as TOTAL_INCARCERATION_ALLPRIOR,
        count(*) as NUM_PREV_INCARCERATION_ALLPRIOR,
        sum(b.INCARCERATION_LEN_DAYS)/count(*) as AVG_INCARCERATION_ALLPRIOR 
        FROM incarceration_len as a join incarceration_len as b on a.ID=b.ID
        WHERE b.END_DATE <= a.END_DATE 
        GROUP BY a.ID, a.PREFIX, a.START_DATE, a.END_DATE
        ''',
        '''
        SELECT
        a.ID, a.PREFIX, a.START_DATE, a.END_DATE,
        sum(b.INCARCERATION_LEN_DAYS) as TOTAL_INCARCERATION_LAST5YR,
        count(*) as NUM_PREV_INCARCERATION_LAST5YR,
        sum(b.INCARCERATION_LEN_DAYS)/count(*) as AVG_INCARCERATION_LAST5YR  
        FROM incarceration_len as a join incarceration_len as b on a.ID=b.ID 
        WHERE julianday(b.END_DATE) >= (julianday(a.END_DATE) - 1825)
        GROUP BY a.ID, a.PREFIX, a.START_DATE, a.END_DATE
        '''
        )

    ft.add_features(database_path, table_names, query)
    print('-- incarceration len features completed --')


def add_countyconviction(database_path=DATABASE_FILENAME):
    '''
    Adding county of conviction
    '''
    table_names = ['countyconviction']
    query = ('''
            SELECT OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX, 
            group_concat(distinct COUNTY_OF_CONVICTION_CODE) as COUNTY_CONVICTION
            from OFNT3CE1 
            where OFFENDER_NC_DOC_ID_NUMBER in (select ID from labels) 
            and COMMITMENT_PREFIX in (select PREFIX from labels) 
            group by OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX
            ''',)
    ft.add_features(database_path, table_names, query)
    print(' -- county of conviction added --')


def add_minmaxterm(database_path=DATABASE_FILENAME):
    '''
    Adding min max term
    '''
    table_names = ['minmaxterm']
    query = ('''
    SELECT OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX, 
    group_concat(distinct SERVING_MIN_OR_MAX_TERM_CODE) AS MINMAXTERM
    from OFNT3CE1
    where OFFENDER_NC_DOC_ID_NUMBER in (select ID from labels) 
    and COMMITMENT_PREFIX in (select PREFIX from labels) 
    group by OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX
    ''',)
    ft.add_features(database_path, table_names, query)
    print( '-- min max term added -- ')


## CREATING INDICIES

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


def create_index_incarcerationlen(database_path=DATABASE_FILENAME):
    '''
    Building an index on incarceration_len table to improve perf
    '''
    create_index = (
    '''
    CREATE INDEX IF NOT EXISTS
    idx_id
    ON incarceration_len(ID)
    ''',
    '''
    CREATE INDEX IF NOT EXISTS
    indx_end_date
    ON incarceration_len(END_DATE)
    ''')

    build_index(database_path, create_index)


## SQL TO PANDAS

def get_table(database_path, query, cols):
    '''
    Generic code to get a table from SQLite3 to Pandas
    '''
    con = sqlite3.connect(database_path)
    SQL_Query = pd.read_sql_query(query, conn)

    df = pd.DataFrame(SQL_Query, columns=cols)
    con.close()
    return df


def get_data_table(filename):
    '''
    Getting tables
    '''
    cols = ['ID','PREFIX','START_DATE','END_DATE','LABEL','INCARCERATION_LEN_DAYS', 'TOTAL_INCARCERATION_ALLPRIOR', 
            'NUM_PREV_INCARCERATION_ALLPRIOR', 'AVG_INCARCERATION_ALLPRIOR', 'TOTAL_INCARCERATION_LAST5YR',
            'NUM_PREV_INCARCERATION_LAST5YR', 'AVG_INCARCERATION_LAST5YR']
    query = '''SELECT * from incarceration_len 
            natural join totcntavg_incarceration_allprior
            natural join totcntavg_incarceration_last5yr 
            '''
    df = get_table(DATABASE_FILENAME, query, cols)
    df.to_csv(filename)
    return df


## DATA CLEANING IN PANDAS

def clean_negative_incarceration_len(df):
    features = ['INCARCERATION_LEN_DAYS','TOTAL_INCARCERATION_ALLPRIOR', 'NUM_PREV_INCARCERATION_ALLPRIOR', 
                'AVG_INCARCERATION_ALLPRIOR','TOTAL_INCARCERATION_LAST5YR', 'NUM_PREV_INCARCERATION_LAST5YR', 
                'AVG_INCARCERATION_LAST5YR']
    
    # create incorrect indicator
    col = 'INCARCERATION_LEN_DAYS'
    filtr = df[col]<0
    if filtr:
        df[col+'_incorrect'] = filtr.astype(int)
    else:
        df[col+'_incorrect'] = 0

    # replace missing with mean
    for col in features:
        df[col] = df[col].where(df[col]>=0)   # replace neg values with nulls
        df = pp.na_fill_col(df, col)
    return df

def clean_county(df):
    # create dummies by every county
    df = create_dummies(df, 'COUNTY_CONVICTION')
    # categories = []
    # df = create_dummies_by_cat(df, 'COUNTY_CONVICTION', categories)
    return df


def clean_minmaxterm(df):
    # replace so that max,min and min,max are the same
    r = {'MAX.TERM:,MIN.TERM:': 'MIN.TERM:,MAX.TERM:'}
    df['MINMAXTERM'] = df['MINMAXTERM'].replace(r)

    # create dummies
    df = create_dummies(df, 'MINMAXTERM')
    return df


## GENERIC CLEANING FUNCTIONS

def replace_missing_with_mean(data_with_missing, data_to_calculate_mean, cols):
    '''
    Replaces null values in dataframe with the mean of the col
    Inputs:
        data_with_missing: pandas df
        data_to_calculate_mean: pandas dataframe
        cols: list of col
    Returns a pandas dataframe with missing values replaced
    '''
    values = {}
    for col in cols:
        values[col] = data_to_calculate_mean[col].mean()
    df = data_with_missing.fillna(value=values)
    return df


def replace_missing_value(df, col, val):
    '''
    Replace null values with val specified
    '''
    values = {col: val}
    df.fillna(value=values, inplace=True)
    return df


def create_dummy_by_cat(df, col, categories):
    for cat in categories:
        filtr = df[col]==cat
        df[col+'_{}'.format(cat)] = filtr.astype(int)
    df.drop(col)

def create_dummies(df, categorical_var):
    '''
    Creates dummy variables from categorical var
    and drops the categorical var
    
    Inputs:
        df: pandas dataframe
        categorical: column name
    Returns a new dataframe with dummy variables added
    '''
    dummy = pd.get_dummies(df[categorical_var],
                           prefix=categorical_var,
                           drop_first=True)
    df.drop(categorical_var, axis=1, inplace=True)
    return df.join(dummy)


if __name__ == '__main__':
    add_incarceration_lens()
    add_countyconviction()
    add_minmaxterm()

