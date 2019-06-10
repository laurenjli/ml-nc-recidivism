# ml-nc-recidivism

This repository contains files to run machine learning models that predict recidivism of an inmate in North Carolina. Our definition of recidivism is when an inmate is released from a period of incarceration with at least one felony sentence returns to prison for another period of incarceration with at least one felony sentence within the time period defined. Currently we will consider a recidivism as an inmate returning within 365 days (or 1 year) of release.


# Dependencies

See requirements.txt.


# Data

The data is provided by North Carolina's Department of Public Safety. To obtain the data processed by jtwalsh0, follow the README on [this](https://github.com/jtwalsh0/ncdoc_data) repository. The .sh that needs to be run is in the folder ncdoc_data. We used the following files: OFNT3DE1.csv, OFNT3CE1.csv, INMT4AA1.csv, INMT4BB1.csv, INMT9CF1.csv, OFNT1BA1.csv, OFNT3BB1.csv, INMT4CA1.csv when creating features for our model. The data was downloaded as of 26 April 2019.

# Feature Generation

We generated our features by creating a sqlite3 database that holds that relevant NC DPS data. We built features on the following: 

* Age (at start of incarceration, end of incarceration, and at first incarceration)

* Gender

* Race

* County of conviction

* Min / Max sentence

* Penalty class code

* Offense qualifier code

* Disciplinary infractions


# Pipeline

The pipeline consists of four main files located in the /source folder:

* pipeline.py: helper functions for preprocessing data, running models, and plotting results

* traintest.py: to create the sqlite3 database, labels, features, and csv train/test sets (create_sqldb.py, features.py, and gettinglabels.py feed into this file)

* config.py: to set the parameters for the models you'd like to run

* main.py: to run the models given the parameters from config.py


Running the code can be done as follows:

1. Download the NC DPS data
2. Configure the config.py file
3. Run 
``` python 
python3 main.py 
```

