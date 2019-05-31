import pandas as pd
import gettinglabels
import sqlite3
import pipeline as pp

DATABASE_FILENAME="../ncdoc_data/data/preprocessed/inmates.db"


## FEATURE GENERATION TABLES ##

def create_ft_table(database_path, table_names, insert_query_list):
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
    build_index(database_path, create_index)

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
    build_index(database_path, create_index)

## ADD FEATURES ##

def add_all_features(database_path = DATABASE_FILENAME):
    '''
    This function creates a data table with all features
    '''
    add_gender_race_age()
    add_ages()
    add_countyconviction()
    add_incarceration_lens()
    add_minmaxterm()
    add_num_sentences()

    #add them all into a sql table called "Data"
    table_names = ['data']
    insert_query = ("""
    SELECT labels.ID, labels.PREFIX, labels.START_DATE, labels.END_DATE, labels.LABEL, 
        t9.INMATE_RACE_CODE, 
        t9.INMATE_GENDER_CODE, 
        t9.AGE_AT_START_DATE, 
        t9.AGE_AT_END_DATE,
        t9.AGE_AT_OFFENSE_START,
        t9.AGE_AT_OFFENSE_END,
        t9.AGE_FIRST_SENTENCE,
        t9.NUM_SENTENCES,
        t9.TOTAL_SENT_ALLPRIOR, 
        t9.NUM_PREV_SENT_ALLPRIOR,
        t9.AVG_SENT_ALLPRIOR,
        t9.TOTAL_SENT_LAST5YR,
        t9.NUM_PREV_SENT_LAST5YR,
        t9.AVG_SENT_LAST5YR,
        t9.INCARCERATION_LEN_DAYS,
        t9.TOTAL_INCARCERATION_ALLPRIOR,
        t9.NUM_PREV_INCARCERATION_ALLPRIOR,
        t9.AVG_INCARCERATION_ALLPRIOR,
        t9.TOTAL_INCARCERATION_LAST5YR, 
        t9.NUM_PREV_INCARCERATION_LAST5YR, 
        t9.AVG_INCARCERATION_LAST5YR,
        t9.MINMAXTERM, 
        t9.COUNTY_CONVICTION
    FROM
    (labels LEFT JOIN
        (inmate_char
        LEFT JOIN
            (age_features
            LEFT JOIN
                (num_sent
                    LEFT JOIN
                    (totcntavg_sentences_allprior
                    LEFT JOIN
                        (totcntavg_sentences_last5yr
                        LEFT JOIN
                            (incarceration_len
                            LEFT JOIN
                                (totcntavg_incarceration_allprior
                                LEFT JOIN 
                                    (totcntavg_incarceration_last5yr
                                    LEFT JOIN
                                        (minmaxterm NATURAL JOIN countyconviction) as t1
                                    ON totcntavg_incarceration_last5yr.ID = t1.ID AND totcntavg_incarceration_last5yr.PREFIX = t1.PREFIX) as t2
                                ON totcntavg_incarceration_allprior.ID = t2.ID AND totcntavg_incarceration_allprior.PREFIX = t2.PREFIX) as t3
                            ON incarceration_len.ID = t3.ID AND incarceration_len.PREFIX = t3.PREFIX) as t4
                        ON totcntavg_sentences_last5yr.ID = t4.ID AND totcntavg_sentences_last5yr.PREFIX = t4.PREFIX) as t5
                    ON totcntavg_sentences_allprior.ID = t5.ID AND totcntavg_sentences_allprior.PREFIX = t5.PREFIX) as t6
                ON num_sent.ID = t6.ID AND num_sent.PREFIX = t6.PREFIX) as t7
            ON age_features.ID = t7.ID AND age_features.PREFIX = t7.PREFIX) as t8
        ON inmate_char.ID = t8.ID AND inmate_char.PREFIX = t8.PREFIX) as t9
    ON labels.ID = t9.ID AND labels.PREFIX = t9.PREFIX)
    """,)
    create_ft_table(database_path, table_names, insert_query)



## Age, race, age

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
    )
    SELECT * from inmate_char
    """,)
    table_names = ['inmate_char']
    create_ft_table(database_path, table_names, query)

    indexq = (
    """
    CREATE INDEX IF NOT EXISTS
    idx_birth
    ON inmate_char(INMATE_BIRTH_DATE)
    """,)
    build_index(database_path=DATABASE_FILENAME, query = indexq)

# More age features

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
    create_ft_table(database_path, table_names, query)

## Number of sentences

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
    create_ft_table(database_path, table_names, query)

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

    create_ft_table(database_path, table_names2, query2)

## Incarceration length

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
    create_ft_table(database_path, table_names, query)
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

    create_ft_table(database_path, table_names, query)

## County of convictions
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
    create_ft_table(database_path, table_names, query)

## Min/max terms

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
    create_ft_table(database_path, table_names, query)




## FEATURE CLEANING ##

def impute_negative_incarceration_len(df):
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

def impute_county(df):
    # create dummies by every county
    df = create_dummies(df, 'COUNTY_CONVICTION')
    # categories = []
    # df = create_dummies_by_cat(df, 'COUNTY_CONVICTION', categories)
    return df


def impute_minmaxterm(df):
    # replace so that max,min and min,max are the same
    r = {'MAX.TERM:,MIN.TERM:': 'MIN.TERM:,MAX.TERM:'}
    df['MINMAXTERM'] = df['MINMAXTERM'].replace(r)

    # create dummies
    df = create_dummies(df, 'MINMAXTERM')
    return df

def impute_race(df, racecol = 'INMATE_RACE_CODE', fill_method = 'Missing'):
    '''
    This function creates a missing binary column and imputes race

    df: dataframe
    fill_method: impute value

    returns dataframe with imputed data
    '''
    # create binary missing column
    # df = pp.missing_col(df, racecol)
    # copy dataframe to impute values then reinsert into df
    cp = df.copy()
    cp.loc[cp[racecol].isna(), racecol] = fill_method
    df[racecol] = cp[racecol]
    return df



if __name__ == '__main__':
    add_all_features()