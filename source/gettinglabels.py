'''
Code to join tables and get labels for NC recidivism data
'''

import sqlite3
import csv

DATABASE_FILENAME="../ncdoc_data/data/preprocessed/inmates.db"

def query_db(query, table_name='data', csv_filename="db_output.csv"):
    '''
    Code to query db from python
    Creates new table to place output and writes in csv file
    Input:
        query: (str) sql query
        table_name: (str) new table name, optional
        csv_filename: output file name, optional
    Returns None
    '''
    con = sqlite3.connect(DATABASE_FILENAME)
    cur = con.cursor()

    rv = []
    
    output = cur.execute(query).fetchall()
    header = get_header(cur)
    
    if output:
        rv.append(header)
        for row in output:
            rv.append(row)

        # write into new table (data)
        cur.execute('DROP TABLE IF EXISTS {}'.format(table_name))
        col_names = []
        for col in header:
            col_names.append(col)
        cur.execute('CREATE TABLE {} ({});'.format(table_name, ",".join(col_names)))
        for row in rv:
            cur.execute('INSERT INTO {} VALUES ({});'.format(table_name,",".join(['?']*len(header))), row)
            
        # write into csv
        with open(csv_filename, 'w') as f:
            csvwriter = csv.writer(f)
            for row in rv:
                csvwriter.writerow(row) 

    con.commit()
    con.close()


def get_header(cursor):
    '''
    Given a cursor object, returns the appropriate header (column names)
    '''
    header = []

    for i in cursor.description:
        s = i[0]
        header.append(s)

    return header
    
def create_labels(time_period, tablename):
    pass


    


if __name__ == '__main__':
    q = "WITH \
    felons_only as (\
    select distinct OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX\
    from OFNT3CE1 \
    where PRIMARY_FELONY/MISDEMEANOR_CD. = 'FELON'), \
    sentence_comp as (\
    select INMATE_DOC_NUMBER as ID, INMATE_COMMITMENT_PREFIX as PREFIX, min(SENTENCE_BEGIN_DATE_FOR_MAX) as start, max(PROJECTED_RELEASE_DATE_PRD, ACTUAL_SENTENCE_END_DATE) as end \
    FROM INMT4BB1 \
    where SENTENCE_BEGIN_DATE_FOR_MAX NOT LIKE '0001%' \
    and SENTENCE_BEGIN_DATE_FOR_MAX NOT LIKE '9999%'\
    and PROJECTED_RELEASE_DATE_PRD NOT LIKE '0001%' \
    and PROJECTED_RELEASE_DATE_PRD NOT LIKE '9999%' \
    and ACTUAL_SENTENCE_END_DATE NOT LIKE '0001%' \
    and ACTUAL_SENTENCE_END_DATE NOT LIKE '9999%' \
    group by INMATE_DOC_NUMBER, INMATE_COMMITMENT_PREFIX),\
    court_commitment as (\
    select OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX, EARLIEST_SENTENCE_EFFECTIVE_DT, NEW_PERIOD_OF_INCARCERATION_FL \
    from OFNT3BB1 \
    where NEW_PERIOD_OF_INCARCERATION_FL = 'Y' \
    where EARLIEST_SENTENCE_EFFECTIVE_DT NOT LIKE '0001%' \
    and EARLIEST_SENTENCE_EFFECTIVE_DT NOT LIKE '9999%'), \
    joined as (\
    select sentence_comp.ID, sentence_comp.PREFIX, min(court_commitment.EARLIEST_SENTENCE_EFFECTIVE_DT, sentence_comp.start) as START_DATE, sentence_comp.end as END_DATE\
    from sentence_comp natural join court_commitment)\
    select felons_only.ID, felons_only.PREFIX, joined.START_DATE, joined.END_DATE \
    from felons_only natural join joined;"

    # current query just joins the two table
    # to exclude sentences that are cancelled/not new incarceration
    # to create labels
    
    query_db(q)


# In sqlite:
# WITH felons_only as (
#     select distinct OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX
#     from OFNT3CE1 where PRIMARY_FELONYMISDEMEANOR_CD LIKE 'FELON'), 
#     sentence_comp as (
#     select INMATE_DOC_NUMBER as ID, INMATE_COMMITMENT_PREFIX as PREFIX, min(SENTENCE_BEGIN_DATE_FOR_MAX) as start, max(PROJECTED_RELEASE_DATE_PRD, ACTUAL_SENTENCE_END_DATE) as end 
#     FROM INMT4BB1 
#     where SENTENCE_BEGIN_DATE_FOR_MAX NOT LIKE '0001%' 
#     and SENTENCE_BEGIN_DATE_FOR_MAX NOT LIKE '9999%'
#     and PROJECTED_RELEASE_DATE_PRD NOT LIKE '0001%' 
#     and PROJECTED_RELEASE_DATE_PRD NOT LIKE '9999%' 
#     and ACTUAL_SENTENCE_END_DATE NOT LIKE '0001%' 
#     and ACTUAL_SENTENCE_END_DATE NOT LIKE '9999%' 
#     group by INMATE_DOC_NUMBER, INMATE_COMMITMENT_PREFIX),
#     court_commitment as (
#     select OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX, EARLIEST_SENTENCE_EFFECTIVE_DT, NEW_PERIOD_OF_INCARCERATION_FL 
#     from OFNT3BB1 
#     where NEW_PERIOD_OF_INCARCERATION_FL = 'Y' 
#     and EARLIEST_SENTENCE_EFFECTIVE_DT NOT LIKE '0001%' 
#     and EARLIEST_SENTENCE_EFFECTIVE_DT NOT LIKE '9999%'), 
#     joined as (
#     select sentence_comp.ID, sentence_comp.PREFIX, min(court_commitment.EARLIEST_SENTENCE_EFFECTIVE_DT, sentence_comp.start) as START_DATE, sentence_comp.end as END_DATE
#     from sentence_comp natural join court_commitment)
#     select felons_only.ID, felons_only.PREFIX, joined.START_DATE, joined.END_DATE
#     from felons_only natural join joined;

