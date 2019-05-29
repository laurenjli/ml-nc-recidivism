'''
Code for creating features
Rachel Ker
'''
import pandas as pd
import gettinglabels
import sqlite3

DATABASE_FILENAME="../ncdoc_data/data/preprocessed/inmates.db"


## FEATURE GENERATION ##

def add_features(database_path, table_names, insert_query_list):
    # create new tables for new features
    for i, query in enumerate(insert_query_list):
        name = table_names[i]
        gettinglabels.query_db(query, args=None, database_path=database_path, table_name=name, new_table=True)
        print("table {} created".format(name))


def add_incarceration_lens(database_path=DATABASE_FILENAME):

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
    add_features(database_path, table_names, query)
    create_index()

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

    add_features(database_path, table_names, query)
    print('-- incarceration len features completed --')



## CREATING INDICIES

def build_index(database_path, query):
    con = sqlite3.connect(database_path)
    cur = con.cursor()
    for q in query:
        cur.execute(q)
        con.commit()
        print('index done')
    con.close()


def create_index(database_path=DATABASE_FILENAME):
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


## DATA CLEANING IN PANDAS ##

def get_table(database_path, query, cols):
    con = sqlite3.connect(database_path)
    SQL_Query = pd.read_sql_query(query, conn)

    df = pd.DataFrame(SQL_Query, columns=cols)
    con.close()
    return df


def get_data_table():
    cols = ['ID','PREFIX','START_DATE','END_DATE','LABEL','INCARCERATION_LEN_DAYS', 'TOTAL_INCARCERATION_ALLPRIOR', 
            'NUM_PREV_INCARCERATION_ALLPRIOR', 'AVG_INCARCERATION_ALLPRIOR', 'TOTAL_INCARCERATION_LAST5YR',
            'NUM_PREV_INCARCERATION_LAST5YR', 'AVG_INCARCERATION_LAST5YR']
    query = '''SELECT * from incarceration_len 
            natural join totcntavg_incarceration_allprior
            natural join totcntavg_incarceration_last5yr 
            '''
    df = get_table(DATABASE_FILENAME, query, cols)
    return df


def deal_with_negative_incarceration_len(df):
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
    df = replace_missing_with_mean(df, df, features)
    return df


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




if __name__ == '__main__':
    add_incarceration_lens()

