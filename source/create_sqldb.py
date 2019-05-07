con = sqlite3.connect(os.path.join(data_dir, "inmates.db"))
cur = con.cursor()

tables = ['INMT4AA1', 'INMT4BB1', 'INMT9CF1',\
          'OFNT1BA1', 'OFNT3BB1', 'OFNT3CE1',\
          'OFNT3DE1', 'INMT4CA1']

for table in tables:
    #print(table)
    file_name = os.path.join(data_dir, "preprocessed/{}.csv".format(table)) 
    col_names = pd.read_csv(file_name, nrows=0).columns
    n_columns = len(col_names)
    col_names = clean_str(', '.join(col_names))
    cur.execute('DROP TABLE IF EXISTS {}'.format(table))
    cur.execute("CREATE TABLE {} ({});".format(table, col_names))
    
    #File contains NULL bytes. That's why I replaced '\0' with ''
    reader = csv.reader(x.replace('\0','') for x in open(file_name))
    for row in reader:
        row = [None if x == '' else x for x in row]
        cur.execute("INSERT INTO {} VALUES ({});".format(table,",".join(['?']*n_columns)), row)

con.commit()
con.close()