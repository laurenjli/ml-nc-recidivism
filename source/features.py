import pandas as pd
import gettinglabels
import sqlite3
import pipeline as pp
import config


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

def create_index_incarcerationlen(database_path=config.DATABASE_FILENAME):
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

def create_labels_indices(database_path=config.DATABASE_FILENAME):
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

def create_OFNT3CE1_indices(database_path=config.DATABASE_FILENAME):
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
    """,
    """
    CREATE INDEX IF NOT EXISTS
    idx_OFNT3CE1_prefix
    ON OFNT3CE1(COMMITMENT_PREFIX)
    """,
    """
    CREATE INDEX IF NOT EXISTS
    idx_OFNT3CE1_id_prefix
    ON OFNT3CE1(OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX)
    """)

    build_index(database_path, create_index)

## ADD FEATURES ##

def add_all_features(database_path = config.DATABASE_FILENAME):
    '''
    This function creates a data table with all features
    '''
    add_gender_race_age()
    add_ages()
    add_countyconviction()
    add_incarceration_lens()
    add_minmaxterm()
    add_num_sentences()
    add_infractions()
    add_offense_penalty()

    #add them all into a sql table called "Data"
    table_names = ['data']
    insert_query = ("""
    SELECT l.ID, l.PREFIX, l.START_DATE, l.END_DATE, l.LABEL, 
        t1.INFRACTIONS,
        t1.INFRACTIONS_UNIQUE,
        t1.INFRACTIONS_GUILTY,
        t1.INFRACTIONS_LAST_INCAR,
        t1.INFRACTIONS_LAST_INCAR_GUILTY,
        t2.PRIMARY_OFFENSE_CODE,
        t2.OFFENSE_QUALIFIER_CODE,
        t2.SENTENCING_PENALTY_CLASS_CODE,
        t3.INMATE_RACE_CODE, 
        t3.INMATE_GENDER_CODE, 
        t3.AGE_AT_START_DATE, 
        t3.AGE_AT_END_DATE,
        t4.AGE_AT_OFFENSE_START,
        t4.AGE_AT_OFFENSE_END,
        t4.AGE_FIRST_SENTENCE,
        t5.TOTAL_SENT_ALLPRIOR, 
        t5.NUM_PREV_SENT_ALLPRIOR,
        t5.AVG_SENT_ALLPRIOR,
        t6.TOTAL_SENT_LAST5YR,
        t6.NUM_PREV_SENT_LAST5YR,
        t6.AVG_SENT_LAST5YR,
        t7.INCARCERATION_LEN_DAYS,
        t8.TOTAL_INCARCERATION_ALLPRIOR,
        t8.NUM_PREV_INCARCERATION_ALLPRIOR,
        t8.AVG_INCARCERATION_ALLPRIOR,
        t9.TOTAL_INCARCERATION_LAST5YR, 
        t9.NUM_PREV_INCARCERATION_LAST5YR, 
        t9.AVG_INCARCERATION_LAST5YR,
        t10.MINMAXTERM, 
        t11.COUNTY_CONVICTION, 
        t12.NUM_SENTENCES
    FROM labels as l
    LEFT JOIN infractions as t1
    ON l.ID = t1.ID AND l.PREFIX = t1.PREFIX
    LEFT JOIN offense_penalty AS t2
    ON l.ID = t2.ID AND l.PREFIX = t2.PREFIX 
    LEFT JOIN inmate_char AS t3
    ON l.ID = t3.ID AND l.PREFIX = t3.PREFIX
    LEFT JOIN age_features as t4
    ON l.ID = t4.ID AND l.PREFIX = t4.PREFIX
    LEFT JOIN totcntavg_sentences_allprior as t5
    ON l.ID = t5.ID AND l.PREFIX = t5.PREFIX
    LEFT JOIN totcntavg_sentences_last5yr as t6
    ON l.ID = t6.ID AND l.PREFIX = t6.PREFIX
    LEFT JOIN incarceration_len as t7
    ON l.ID = t7.ID AND l.PREFIX = t7.PREFIX
    LEFT JOIN totcntavg_incarceration_allprior as t8
    ON l.ID = t8.ID AND l.PREFIX = t8.PREFIX
    LEFT JOIN totcntavg_incarceration_last5yr as t9
    ON l.ID = t9.ID AND l.PREFIX = t9.PREFIX
    LEFT JOIN minmaxterm as t10
    ON l.ID = t10.ID AND l.PREFIX = t10.PREFIX
    LEFT JOIN countyconviction as t11
    ON l.ID = t11.ID AND l.PREFIX = t11.PREFIX
    LEFT JOIN num_sent as t12
    ON l.ID = t12.ID AND l.PREFIX = t12.PREFIX)
    """,)
    create_ft_table(database_path, table_names, insert_query)

## Age, race, age

def add_gender_race_age(database_path=config.DATABASE_FILENAME):
    create_labels_indices()

    query = (
    """
    WITH null_bday as (
        SELECT INMATE_DOC_NUMBER, INMATE_GENDER_CODE,INMATE_RACE_CODE,
        CASE WHEN (INMATE_BIRTH_DATE LIKE '0001%') THEN NULL
        ELSE INMATE_BIRTH_DATE END AS INMATE_BIRTH_DATE
        FROM
        INMT4AA1
    )
    SELECT ID, PREFIX, START_DATE, END_DATE,INMATE_GENDER_CODE, INMATE_RACE_CODE, INMATE_BIRTH_DATE,
    CASE WHEN (julianday(START_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 < 0
    THEN NULL
    ELSE (julianday(START_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 END as AGE_AT_START_DATE, 
    CASE WHEN (julianday(END_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 < 0
    THEN NULL
    ELSE (julianday(END_DATE) - julianday(INMATE_BIRTH_DATE))/365.0 END as AGE_AT_END_DATE
    FROM
    labels LEFT JOIN null_bday
    ON null_bday.INMATE_DOC_NUMBER = labels.ID

    """,)
    table_names = ['inmate_char']
    create_ft_table(database_path, table_names, query)

    indexq = (
    """
    CREATE INDEX IF NOT EXISTS
    idx_birth
    ON inmate_char(INMATE_BIRTH_DATE)
    """,)
    build_index(database_path=config.DATABASE_FILENAME, query = indexq)

# More age features

def add_ages(database_path=config.DATABASE_FILENAME):
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

def add_num_sentences(database_path=config.DATABASE_FILENAME):
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

def add_incarceration_lens(database_path=config.DATABASE_FILENAME):
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
def add_countyconviction(database_path=config.DATABASE_FILENAME):
    '''
    Adding county of conviction
    '''
    table_names = ['countyconviction']
    query = ('''
            SELECT OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX, 
            group_concat(distinct COUNTY_OF_CONVICTION_CODE) as COUNTY_CONVICTION
            FROM labels as l
            LEFT JOIN OFNT3CE1 as o
            ON l.ID == o.OFFENDER_NC_DOC_ID_NUMBER 
            AND  l.PREFIX == o.COMMITMENT_PREFIX
            group by OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX
            ''',)
    create_ft_table(database_path, table_names, query)


# def get_unique_county(database_path=config.DATABASE_FILENAME):
#     con = sqlite3.connect(database_path)
#     cur = con.cursor()
#     q = '''SELECT DISTINCT
#     COUNTY_OF_CONVICTION_CODE
#     FROM OFNT3CE1
#     '''
#     cur.execute(q)
#     output = cur.fetchall()
#     con.close()

#     rv=[]
#     for tup in output:
#         county = tup[0]
#         if county=='COUNTY_OF_CONVICTION_CODE':
#             continue
#         else:
#             rv.append(county)
#     return rv

## Min/max terms

def add_minmaxterm(database_path=config.DATABASE_FILENAME):
    '''
    Adding min max term
    '''
    table_names = ['minmaxterm']
    query = ('''
    SELECT OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX, 
    group_concat(distinct SERVING_MIN_OR_MAX_TERM_CODE) AS MINMAXTERM
    FROM labels as l
    LEFT JOIN OFNT3CE1 as o
    ON l.ID == o.OFFENDER_NC_DOC_ID_NUMBER 
    AND  l.PREFIX == o.COMMITMENT_PREFIX
    group by OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX
    ''',)
    create_ft_table(database_path, table_names, query)


def add_infractions(database_path=config.DATABASE_FILENAME):
    '''
    Adding infractions
    '''
    table_names = ['infractions']
    query = ('''
    SELECT 
        l.ID, 
        l.PREFIX, 
        COUNT(DISCIPLINARY_INFRACTION_DATE) as INFRACTIONS,
        COUNT(DISTINCT DISCIPLINARY_INFRACTION_DATE) as INFRACTIONS_UNIQUE,
        SUM(CASE WHEN  i.DISCIINFRACTION_VERDICT_CODE == 'GUILTY' THEN 1 ELSE 0 END)  as INFRACTIONS_GUILTY,
        SUM(CASE WHEN  i.DISCIPLINARY_INFRACTION_DATE <= l.START_DATE THEN 1 ELSE 0 END) as INFRACTIONS_LAST_INCAR,
        SUM(CASE WHEN  i.DISCIPLINARY_INFRACTION_DATE <= l.START_DATE AND i.DISCIINFRACTION_VERDICT_CODE == 'GUILTY' THEN 1 ELSE 0 END) as INFRACTIONS_LAST_INCAR_GUILTY
    FROM labels as l
    LEFT JOIN INMT9CF1 as i
    ON l.ID == i.INMATE_DOC_NUMBER
    WHERE i.DISCIINFRACTION_VERDICT_CODE == 'GUILTY'
    AND l.END_DATE >= i.DISCIPLINARY_INFRACTION_DATE
    GROUP BY l.ID, l.PREFIX
    ''',)
    create_ft_table(database_path, table_names, query)


## County of convictions
def add_offense_penalty(database_path=config.DATABASE_FILENAME):
    '''
    Adding primary offense, offense qualifier and sentencing penaltyn
    '''
    table_names = ['offense_penalty']
    query = ('''
    SELECT
        l.ID, 
        l.PREFIX,
        GROUP_CONCAT(distinct REPLACE(o.PRIMARY_OFFENSE_CODE, ',', '')) as PRIMARY_OFFENSE_CODE,
        GROUP_CONCAT(distinct REPLACE(o.OFFENSE_QUALIFIER_CODE, ',', '')) as OFFENSE_QUALIFIER_CODE,
        GROUP_CONCAT(distinct REPLACE(o.SENTENCING_PENALTY_CLASS_CODE, ',', '')) as SENTENCING_PENALTY_CLASS_CODE
    FROM labels as l
    LEFT JOIN OFNT3CE1 as o
    ON l.ID == o.OFFENDER_NC_DOC_ID_NUMBER 
    AND  l.PREFIX == o.COMMITMENT_PREFIX
    GROUP BY l.ID, l.PREFIX
    ''',)
    create_ft_table(database_path, table_names, query)





## FEATURE CLEANING ## THIS CAN BE DELETED, TRUE?

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