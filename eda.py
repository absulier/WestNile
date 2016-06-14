import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv('data/train.csv')
weather=pd.read_csv('data/weather.csv')
spray=pd.read_csv('data/spray.csv')

df.describe()
weather.describe()
spray.describe()

df.columns
weather.columns
spray.columns

df.drop('Address',axis=1,inplace=True)
df

df2 = df.groupby(by=["Trap"], as_index=False)
df2 = df2.agg({'NumMosquitos': np.count_nonzero,
               'WnvPresent': np.count_nonzero,
                }).dropna()

df2

plt.hist(df2.NumMosquitos)
plt.show()

plt.scatter(df2.NumMosquitos,df2.WnvPresent)
plt.show()

present=[]
for i in df2.WnvPresent:
    if i>0:
        present.append(1)
    else:
        present.append(0)

df2['present']=present

plt.scatter(df2.NumMosquitos,df2.present)
plt.show()
