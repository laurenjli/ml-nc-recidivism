data_dir = "../ncdoc_data/data/preprocessed"
con = sqlite3.connect(os.path.join(data_dir, "inmates.db"))
cur = con.cursor()

tables =  ['INMT4AA1', 'INMT4BB1', 'INMT9CF1',\
          'OFNT1BA1', 'OFNT3BB1', 'OFNT3CE1',\
          'OFNT3DE1', 'INMT4CA1']


for table in tables:
    print(table)
    start = time. time()
    file_name = os.path.join(data_dir, "{}.csv".format(table)) 
    df = pd.read_csv(file_name)
    cur.execute('DROP TABLE IF EXISTS {}'.format(table))
    df.to_sql(table, con, index=False)
    end = time. time()
    print(end - start)
    
con.commit()
con.close()
