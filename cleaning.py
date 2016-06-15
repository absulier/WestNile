import pandas as pd
import numpy as np

#read in traning data
df=pd.read_csv('data/train2.csv').drop(['Address','AddressAccuracy','AddressNumberAndStreet'],axis=1)

#creating a score for each block based on the percentage of
#samples found on that block with WNV
df2 = df.groupby(by=["Block"], as_index=False)
df2 = df2.agg({'WnvPresent': np.mean,
            }).dropna()
#creating a score for each block based on the percentage of
#samples found on that block with WNV
df3 = df.groupby(by=["Street"], as_index=False)
df3 = df3.agg({'WnvPresent': np.mean,
            }).dropna()
#creating a score for each block based on the percentage of
#samples found on that block with WNV
df4 = df.groupby(by=["Trap"], as_index=False)
df4 = df4.agg({'WnvPresent': np.mean,
            }).dropna()
#creating a score for each week based on the percentage of
#samples found in that month with WNV
month=[]
for i in df.Date:
    if int(i[8:10]) < 8:
        month.append(i[5:7]+'.1')
    elif int(i[8:10]) < 16:
        month.append(i[5:7]+'.2')
    elif int(i[8:10]) < 24:
        month.append(i[5:7]+'.3')
    else:
        month.append(i[5:7]+'.4')
df['month']=month
df5 = df.groupby(by=["month"], as_index=False)
df5 = df5.agg({'WnvPresent': np.mean,
            }).dropna()

#puts overwrites the block, street, trap, and week columns with their respecitive scores
df.Block=df.merge(df2,on='Block',how='left').WnvPresent_y
df.Street=df.merge(df3,on='Street',how='left').WnvPresent_y
df.Trap=df.merge(df4,on='Trap',how='left').WnvPresent_y
df.month=df.merge(df5,on='month',how='left').WnvPresent_y

#Creates dummy variables and merges them into the dataframe, deletes original speciec
df=df.merge(pd.get_dummies(df.Species),left_index=True,right_index=True).drop('Species',axis=1)

#renames columns to make them more workable
df.columns=['date','block','street','trap','lat','lon','num_mos','wnv','week','s_err','s_pip','s_p/r','s_res','s_sal','s_tar','s_ter']

#Read in weather data
we=pd.read_csv('data/weather.csv')
we=we[we['Station']==1].drop(['Station','CodeSum','Water1'],axis=1)

#convert missing values to 0 and turn to floats
def to_float(x):
    if 'T' in str(x) or 'M' in str(x):
        return 0
    else:
        return float(x)
for i in we.drop('Date',axis=1).columns:
    we[i]=we[i].apply(to_float)

#print the clean data
pd.DataFrame.to_csv(df,'data_clean/df.csv',index=False)
pd.DataFrame.to_csv(we,'data_clean/we.csv',index=False)
pd.DataFrame.to_csv(df2,'data_clean/df2.csv',index=False)
pd.DataFrame.to_csv(df3,'data_clean/df3.csv',index=False)
pd.DataFrame.to_csv(df4,'data_clean/df4.csv',index=False)
pd.DataFrame.to_csv(df5,'data_clean/df5.csv',index=False)
