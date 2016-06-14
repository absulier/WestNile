import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
%matplotlib inline

#read in data
df=pd.read_csv('data/train.csv')
weather=pd.read_csv('data/weather.csv')
spray=pd.read_csv('data/spray.csv')

#check out data
df.head()
weather.head()
spray.head()

#briefly look at sumamry stats
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

#look to see if data is fairly normal
normaltest(df[['Block','Latitude','Longitude','AddressAccuracy','NumMosquitos','WnvPresent']], axis=0)
normaltest(weather[['Station','Tmax','Tmin','DewPoint','ResultSpeed','ResultDir']],axis=0)
normaltest(spray[['Latitude','Longitude']],axis=0)

#redundant columns, also not useful
df.drop(['Address','AddressNumberAndStreet'],axis=1,inplace=True)

#Look at the mosquitos sorted by trap
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

#creats simple binary for if virus was present from total number of instances
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

#Looking at the time of year the virus shows up
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df3=df[['WnvPresent']]

#Plots if WVN was present on that day, note, we do not have data for even numbered years
df3[df3['WnvPresent']==1]
df3.plot(y='WnvPresent')

#Sorting by block to see percentages of present virus on each block
df4 = df.groupby(by=["Block"], as_index=False)
df4 = df4.agg({'NumMosquitos': np.mean,
               'WnvPresent': np.mean,
                }).dropna()


#plots out the percent of samples that had WNV on each block
df4.plot.bar(x='Block',y='WnvPresent')

#plots out the percent of samples that had WNV on each block sorted
df4.sort_values(by='WnvPresent').plot.bar(x='Block',y='WnvPresent')
