'''
Code to join tables and get labels for NC recidivism data
'''

import sqlite3
import csv

DATABASE_FILENAME="../ncdoc_data/inmates.db"

def query_db(query, csv_filename="db_output.csv"):
    '''
    Code to query db from python
    Input:
        query: (str) sql query
        csv_filename: output (optional)
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

        with open(csv_filename, 'w') as f:
            csvwriter = csv.writer(f)
            for row in rv:
                csvwriter.writerow(row) 

    con.close()
    return tuple(rv)


def get_header(cursor):
    '''
    Given a cursor object, returns the appropriate header (column names)
    '''
    header = []

    for i in cursor.description:
        s = i[0]
        header.append(s)

    return header
    


if __name__ == '__main__':
    q = "WITH \
    sentence_comp(INMATE_DOC_NUMBER,INMATE_COMMITMENT_PREFIX,BEGIN_DATE,RELEASE_DATE) as (\
    select INMATE_DOC_NUMBER, INMATE_COMMITMENT_PREFIX, min(SENTENCE_BEGIN_DATE_FOR_MAX), max(PROJECTED_RELEASE_DATE_PRD, ACTUAL_SENTENCE_END_DATE) \
    FROM INMT4BB1 where SENTENCE_BEGIN_DATE_FOR_MAX != '0001-01-01'and PROJECTED_RELEASE_DATE_PRD != '9999-01-03' and ACTUAL_SENTENCE_END_DATE != '9999-01-03' \
    group by INMATE_DOC_NUMBER, INMATE_COMMITMENT_PREFIX),\
    court_commitment(OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX, EARLIEST_SENTENCE_EFFECTIVE_DT, NEW_PERIOD_OF_INCARCERATION_FL) as (\
    select OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX, EARLIEST_SENTENCE_EFFECTIVE_DT, NEW_PERIOD_OF_INCARCERATION_FL from OFNT3BB1)\
    select * from sentence_comp join court_commitment where sentence_comp.INMATE_DOC_NUMBER == court_commitment.OFFENDER_NC_DOC_ID_NUMBER \
    and sentence_comp.INMATE_COMMITMENT_PREFIX == court_commitment.COMMITMENT_PREFIX;"
    query_db(q)


# In sqlite:
##   WITH Sentence_comp(INMATE_DOC_NUMBER,INMATE_COMMITMENT_PREFIX,BEGIN_DATE,RELEASE_DATE) as (
##   select INMATE_DOC_NUMBER, INMATE_COMMITMENT_PREFIX, min(SENTENCE_BEGIN_DATE_FOR_MAX), max(PROJECTED_RELEASE_DATE_PRD, ACTUAL_SENTENCE_END_DATE)
##   FROM INMT4BB1 where SENTENCE_BEGIN_DATE_FOR_MAX != '0001-01-01'and PROJECTED_RELEASE_DATE_PRD != '9999-01-03' and ACTUAL_SENTENCE_END_DATE != '9999-01-03' 
##   group by INMATE_DOC_NUMBER, INMATE_COMMITMENT_PREFIX),
##   court_commitment(OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX, EARLIEST_SENTENCE_EFFECTIVE_DT, NEW_PERIOD_OF_INCARCERATION_FL) as (
##   select OFFENDER_NC_DOC_ID_NUMBER, COMMITMENT_PREFIX, EARLIEST_SENTENCE_EFFECTIVE_DT, NEW_PERIOD_OF_INCARCERATION_FL from OFNT3BB1)
##   select * from sentence_comp join court_commitment where sentence_comp.INMATE_DOC_NUMBER == court_commitment.OFFENDER_NC_DOC_ID_NUMBER 
##   and sentence_comp.INMATE_COMMITMENT_PREFIX == court_commitment.COMMITMENT_PREFIX limit 10;

