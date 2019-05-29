'''
Code for creating features
Rachel Ker
'''

import gettinglabels
from create_sqldb import create_db

import sqlite3

DATABASE_FILENAME="../ncdoc_data/data/preprocessed/inmates.db"

def add_first_feature(database_path=DATABASE_FILENAME):
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
    print('first feature created')


def add_new_features(database_path, new_features_list, insert_query_list):

    # connect to database to add more features
    con = sqlite3.connect(database_path)
    cur = con.cursor()

    # add columns for new features
    create_new_col = ''''''
    for new in new_features_list:
        cur.execute('ALTER TABLE data ADD COLUMN {}'.format(new))
        print('added column {}'.format(new))
    
    # insert values into new columns
    for q in insert_query_list:
        cur.execute(q)
        print('insert done')
    
    con.commit()
    con.close()


def add_incarceration_lens():

    new_features = ['TOTAL_INCARCERATION_ALLPRIOR', 'NUM_PREV_INCARCERATION_ALLPRIOR', 'AVG_INCARCERATION_ALLPRIOR',
                    'TOTAL_INCARCERATION_LAST5YR', 'NUM_PREV_INCARCERATION_LAST5YR', 'AVG_INCARCERATION_LAST5YR',
                    'RELATIVE_INCARCERATION_LEN_DAYS']

    query = (
        '''
        INSERT INTO data(TOTAL_INCARCERATION_ALLPRIOR, NUM_PREV_INCARCERATION_ALLPRIOR, AVG_INCARCERATION_ALLPRIOR)
        SELECT 
        sum(b.INCARCERATION_LEN_DAYS) as TOTAL_INCARCERATION_ALLPRIOR,
        count(*) as NUM_PREV_INCARCERATION_ALLPRIOR,
        sum(b.INCARCERATION_LEN_DAYS)/count(*) as AVG_INCARCERATION_ALLPRIOR 
        FROM data as a join data as b on a.ID=b.ID
        WHERE b.END_DATE <= a.END_DATE 
        GROUP BY a.ID, a.PREFIX, a.START_DATE, a.END_DATE
        ''',
        '''
        INSERT INTO data(TOTAL_INCARCERATION_LAST5YR, NUM_PREV_INCARCERATION_LAST5YR, AVG_INCARCERATION_LAST5YR)
        SELECT
        sum(b.INCARCERATION_LEN_DAYS) as TOTAL_INCARCERATION_LAST5YR,
        count(*) as NUM_PREV_INCARCERATION_LAST5YR,
        sum(b.INCARCERATION_LEN_DAYS)/count(*) as AVG_INCARCERATION_LAST5YR  
        FROM data as a join data as b on a.ID=b.ID 
        WHERE julianday(b.END_DATE) >= (julianday(a.END_DATE) - 1825)
        GROUP BY a.ID, a.PREFIX, a.START_DATE, a.END_DATE
        ''',
        '''
        INSERT INTO data(RELATIVE_INCARCERATION_LEN_DAYS)
        SELECT 
        INCARCERATION_LEN_DAYS - (SELECT
        sum(b.INCARCERATION_LEN_DAYS)/count(*) as mean
        FROM data as a, data as b
        WHERE b.END_DATE <= a.END_DATE) as RELATIVE_INCARCERATION_LEN_DAYS
        from data
        '''
        )

    add_new_features(DATABASE_FILENAME, new_features, query)
    print('created incarceration len features')
    


if __name__ == '__main__':
    add_first_feature()
    add_incarceration_lens()

