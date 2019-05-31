# ml-nc-recidivism

This repository contains files to run machine learning models that predict recidivism of an inmate in North Carolina. Our definition of recidivism is when an inmate is released from a period of incarceration with at least one felony sentence returns to prison for another period of incarceration with at least one felony sentence within the time period defined. Currently we will consider a recidivism as an inmate returning within 365 days (or 1 year) of release.

# Dependencies

--python 3.x

--pandas

--numpy

--sklearn

--os

--sqlite3

--warnings


# Data

The data is provided by North Carolina's Department of Public Safety. To obtain the data processed by jtwalsh0, follow the README on [this](https://github.com/jtwalsh0/ncdoc_data) repository. The .sh that needs to be run is in the folder ncdoc_data. We used the following files: OFNT3DE1.csv, OFNT3CE1.csv, INMT4AA1.csv, INMT4BB1.csv, INMT9CF1.csv, OFNT1BA1.csv, OFNT3BB1.csv, INMT4CA1.csv when creating features for our model. The data was downloaded as of 26 April 2019.

# Feature Generation

# Pipeline

# Example Pipeline Run

