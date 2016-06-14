import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest
%matplotlib inline

df=pd.read_csv('data/train.csv')
weather=pd.read_csv('data/weather.csv')
spray=pd.read_csv('data/spray.csv')

df.head()
weather.head()
spray.head()

df.describe()
weather.describe()
spray.describe()

#There is a mix of both categorical and numeric data in the main dataframe
df.dtypes
#The weather seems to be mostly cateogrical and dummy variable. This will need to be cleaned,
#especially since a lot of variables that appear to be either numberic of dummy
#variables are coming through as objects
weather.dtypes
#This iseems pretty straight forward of just the time and location of each spraying
spray.dtypes


normaltest(df[['Block','Latitude','Longitude','AddressAccuracy','NumMosquitos','WnvPresent']], axis=0)
normaltest(weather[['Station','Tmax','Tmin','DewPoint','ResultSpeed','ResultDir']],axis=0)
normaltest(spray[['Latitude','Longitude']],axis=0)

df.drop('Address',axis=1,inplace=True)

df2 = df.groupby(by=["Trap"], as_index=False)
df2 = df2.agg({'NumMosquitos': np.count_nonzero,
               'WnvPresent': np.count_nonzero,
                }).dropna()

#Histogram shows the frequencies of total number of mosquitos per trap
plt.hist(df2.NumMosquitos)
plt.show()

#X axis shows number of mosquitos found in Trap
#Y axis shows number of incidents of west nile
plt.scatter(df2.NumMosquitos,df2.WnvPresent)
plt.show()

present=[]
for i in df2.WnvPresent:
    if i>0:
        present.append(1)
    else:
        present.append(0)
df2['present']=present

#X axis is number of mosquiots in trap
#Y axis is if west nile is present or not
plt.scatter(df2.NumMosquitos,df2.present)
plt.show()
