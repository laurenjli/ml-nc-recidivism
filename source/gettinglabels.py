'''
Code to join tables and get labels for NC recidivism data
'''

import sqlite3
import csv

DATABASE_FILENAME="../ncdoc_data/data/preprocessed/inmates.db"

def query_db(query, args, database_path=DATABASE_FILENAME, table_name='data', new_table=True, csv_filename=None):
    '''
    Code to query db from python
    Creates new table to place output and writes in csv file
    Input:
        query: (str) sql query
        args: a tuple of args
        table_name: (str) new table name, optional
        csv_filename: output file name, optional
    Returns None
    '''
    con = sqlite3.connect(database_path)
    cur = con.cursor()

    rv = []
    
    if args:
        output = cur.execute(query, args).fetchall()
    else:
        output = cur.execute(query).fetchall()
    header = get_header(cur)
    
    if output:
        # write into new table (data)
        if new_table:
            cur.execute('DROP TABLE IF EXISTS {}'.format(table_name))
            col_names = []
            for col in header:
                col_names.append(col)
            cur.execute('CREATE TABLE {} ({});'.format(table_name, ",".join(col_names)))
        for row in output:
            cur.execute('INSERT INTO {} VALUES ({});'.format(table_name,",".join(['?']*len(header))), row)
            
        # write into csv if csv_filename is not None
        if csv_filename:
            rv.append(header)
            for row in output:
                rv.append(row)

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
    
def create_labels(database_path=DATABASE_FILENAME, time_period = 365.0, default_max = 10000.0, table_name = 'labels'):
    '''
    This function creates a new relation in the database with labels for the given time_period.

    time_period: in days (float)
    default_max: default max value to use for rows that do not have a next incarceration (float)
    table_name: name for new table

    returns: none
    '''

    query = "WITH \
    felons_only as (\
    select distinct OFFENDER_NC_DOC_ID_NUMBER as ID, COMMITMENT_PREFIX as PREFIX\
    from OFNT3CE1 \
    where PRIMARY_FELONYMISDEMEANOR_CD = 'FELON'), \
    sentence_comp as (\
    select INMATE_DOC_NUMBER as ID, INMATE_COMMITMENT_PREFIX as PREFIX, min(SENTENCE_BEGIN_DATE_FOR_MAX) as start, max(ACTUAL_SENTENCE_END_DATE) as end_actual, max(PROJECTED_RELEASE_DATE_PRD) as end_proj \
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
    and EARLIEST_SENTENCE_EFFECTIVE_DT NOT LIKE '0001%' \
    and EARLIEST_SENTENCE_EFFECTIVE_DT NOT LIKE '9999%'), \
    joined as (\
    select sentence_comp.ID, sentence_comp.PREFIX, min(court_commitment.EARLIEST_SENTENCE_EFFECTIVE_DT, sentence_comp.start) as START_DATE, min(sentence_comp.end_actual, sentence_comp.end_proj) as END_DATE\
    from sentence_comp natural join court_commitment),\
    final as ( \
    select felons_only.ID, felons_only.PREFIX, joined.START_DATE, joined.END_DATE \
    from felons_only natural join joined \
    group by felons_only.ID, felons_only.PREFIX \
    order by joined.START_DATE) \
    select ID, PREFIX, START_DATE, END_DATE, (case when coalesce((select julianday(START_DATE) \
        from final as t2 \
        where t1.ID = t2.ID \
        and date(t1.END_DATE) <= date(t2.START_DATE) \
        order by date(t2.START_DATE) \
        limit 1 \
    ) - julianday(END_DATE), ?) <= ? then 1 else 0 end) as LABEL \
    from final as t1;"

    args = (default_max, time_period)

    query_db(query, args, database_path, table_name, new_table=True)
    print('label created')
    


if __name__ == '__main__':
    create_labels(database_path = DATABASE_FILENAME,
                  time_period = 365.0, 
                  default_max = 10000.0, 
                  table_name = 'labels')


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
#     from sentence_comp natural join court_commitment),
#     final as (
#     select felons_only.ID, felons_only.PREFIX, joined.START_DATE, joined.END_DATE
#     from felons_only natural join joined 
#     group by felons_only.ID, felons_only.PREFIX
#     order by joined.START_DATE)
#     select ID, PREFIX, START_DATE, END_DATE, (case when coalesce((select julianday(START_DATE)
#         from final as t2
#         where t1.ID = t2.ID
#         and date(t1.END_DATE) <= date(t2.START_DATE)
#         order by date(t2.START_DATE)
#         limit 1
#     ) - julianday(END_DATE), 100000) < 365.0 then 1 else 0 end) as LABEL
#     from final as t1;

