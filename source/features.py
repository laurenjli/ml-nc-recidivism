import pandas as pd
import gettinglabels
import sqlite3
import pipeline as pp

DATABASE_FILENAME="../ncdoc_data/data/preprocessed/inmates.db"


## FEATURE GENERATION TABLES ##

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



## ADD FEATURES ##

## Age, race, age

def add_gender_race_age(database_path=DATABASE_FILENAME):
    query = """
    WITH inmate_char as (
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
    ),
    age_first as(
        SELECT ID, (julianday(min(START_DATE)) - julianday(min(INMATE_BIRTH_DATE)))/365.0 as AGE_FIRST_SENTENCE
        FROM inmate_char
        group by ID
    ),
    age_offense as (
        SELECT inmate_char.ID as ID, inmate_char.PREFIX as PREFIX, OFNT3CE1.min_d as OFFENSE_START, OFNT3CE1.max_d as OFFENSE_END,
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

    ),
    prev as (
        SELECT ID, PREFIX, ROW_NUMBER() OVER (PARTITION BY ID ORDER BY START_DATE) -1 as PREV_INCARC
        FROM labels
        ORDER BY START_DATE
    )
    SELECT * 
    FROM (inmate_char natural join age_first) as t1 natural join 
    (age_offense natural join prev) as t2
    limit 5;
    """
    table_names = ['inmate_char']
    add_features(database_path, table_names, query)

## Number of sentences

def add_num_sentences():
    query = """
    WITH sent as (
        SELECT OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX, count(SENTENCE_COMPONENT_NUMBER) as NUM_SENTENCES
        FROM OFNT3CE1 
        GROUP BY OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX
    )
    SELECT *
    FROM labels natural join sent
    """
    table_names = ['num_sent']
    add_features(database_path, table_names, query)

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
        WHERE julianday(b.END_DATE) >= (julianday(a.END_DATE) - 1825) and julianday(b.END_DATE) <= (julianday(a.END_DATE)
        GROUP BY a.ID, a.PREFIX, a.START_DATE, a.END_DATE
        '''
        )

    add_features(database_path, table_names2, query2)

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
    add_features(database_path, table_names, query)
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

    add_features(database_path, table_names, query)
    print('-- incarceration len features completed --')

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
    add_features(database_path, table_names, query)
    print(' -- county of conviction added --')

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
    add_features(database_path, table_names, query)
    print( '-- min max term added -- ')


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


if __name__ == '__main__':

    # add all features tables
    add_incarceration_lens()
    add_countyconviction()
    add_minmaxterm()

    # create final data table with all features