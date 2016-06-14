import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
%matplotlib inline

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



#basic model, LDA

#read in test data and then clean/transform on the same scores used for training
test=pd.read_csv('data/test.csv').drop(['Address','AddressAccuracy','AddressNumberAndStreet'],axis=1)
month=[]
for i in train.Date:
    if int(i[8:10]) < 8:
        month.append(i[5:7]+'.1')
    elif int(i[8:10]) < 16:
        month.append(i[5:7]+'.2')
    elif int(i[8:10]) < 24:
        month.append(i[5:7]+'.3')
    else:
        month.append(i[5:7]+'.4')
test['month']=month
test.Block=test.merge(df2,on='Block',how='left').WnvPresent_y
test.Street=test.merge(df3,on='Street',how='left').WnvPresent_y
test.Trap=test.merge(df4,on='Trap',how='left').WnvPresent_y
test.month=test.merge(df5,on='month',how='left').WnvPresent_y
test=test.merge(pd.get_dummies(test.Species),left_index=True,right_index=True).drop('Species',axis=1)
test.columns=['date','block','street','trap','lat','lon','num_mos','wnv','week','s_pip','s_p/r','s_res','s_sal','s_ter']

#Some species dont appear in test data, so just adding empty dummy variables to
#make sure test and train has matching features
s_=[]
for i in range(len(train)):
    s_.append(0)
test['s_err'],test['s_tar']=s_,s_
test=test[['date','block','street','trap','lat','lon','num_mos','wnv','week','s_err','s_pip','s_p/r','s_res','s_sal','s_tar','s_ter']]

#declare test target and data
X_test=test.drop(['wnv','date'],axis=1)
y_test=test['wnv']

#running the train and test data in LDA
X=df.drop(['wnv','date'],axis=1)
y=df['wnv']
lda_classifier = LDA(n_components=2)
lda_x_axis = lda_classifier.fit(X, y).transform(X)

lda_classifier.score(X_test, y_test, sample_weight=None)
y_pred=lda_classifier.predict_proba(X_test)
proba=pd.DataFrame(y_pred)[1]
proba.mean()

#play with the predication threshold to see falsenegative/positive trade off
y_pred2=[]
for i in proba:
    if i>.06:
        y_pred2.append(1)
    else:
        y_pred2.append(0)

#(true negative) (false positive)
#(false negative) (true positive)
print confusion_matrix(y_test,y_pred2)

dt = DecisionTreeClassifier(class_weight='balanced')
dt.fit(X,y)
dt.score(X_test,y_test)

rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X,y)
rf.score(X_test,y_test)
