import pandas as pd
from sklearn.lda import LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report as skcr
from sklearn.metrics import roc_curve as skrc
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn
%matplotlib inline


#read in train data and dataframes for feature scores
df=pd.read_csv('data_clean/df.csv')
df2=pd.read_csv('data_clean/df2.csv')
df3=pd.read_csv('data_clean/df3.csv')
df4=pd.read_csv('data_clean/df4.csv')
df5=pd.read_csv('data_clean/df5.csv')
we=pd.read_csv('data_clean/we.csv')

#read in test data and then clean/transform on the same scores used for training
test=pd.read_csv('data/test.csv').drop(['Address','AddressAccuracy','AddressNumberAndStreet'],axis=1)
#make a new column 'month' that has month by quarter
month=[]
for i in test.Date:
    if int(i[8:10]) < 8:
        month.append(i[5:7]+'.1')
    elif int(i[8:10]) < 16:
        month.append(i[5:7]+'.2')
    elif int(i[8:10]) < 24:
        month.append(i[5:7]+'.3')
    else:
        month.append(i[5:7]+'.4')
test['month']=month

#merge the scores for the features into the dataframe, replace the original
#values with the scores
test.Block=test.merge(df2,on='Block',how='left').WnvPresent_y
test.Street=test.merge(df3,on='Street',how='left').WnvPresent_y
test.Trap=test.merge(df4,on='Trap',how='left').WnvPresent_y
to_float=lambda x : float(x)
test.month=test.month.apply(to_float)
test.month=test.merge(df5,on='month',how='left').WnvPresent_y
test=test.merge(pd.get_dummies(test.Species),left_index=True,right_index=True).drop('Species',axis=1)
test.columns=['date','block','street','trap','lat','lon','num_mos','wnv','week','s_pip','s_p/r','s_res','s_sal','s_ter']

#Some species dont appear in test data, so just adding empty dummy variables to
#make sure test and train has matching features
s_=[]
for i in range(len(test)):
    s_.append(0)
test['s_err'],test['s_tar']=s_,s_
test=test[['date','block','street','trap','lat','lon','num_mos','wnv','week','s_err','s_pip','s_p/r','s_res','s_sal','s_tar','s_ter']]

#read in weather data
we=pd.read_csv('data_clean/we.csv')
we =we.rename(columns={'Date':'date'})

#merge we onto the test and train
df=df.merge(we, how='left', on='date')
test=test.merge(we, how='left', on='date')

#declare test target and data
X_test=test.drop(['wnv','date'],axis=1)
y_test=test['wnv']

#build train sets
X=df.drop(['wnv','date'],axis=1)
y=df['wnv']

#running the train and test data in LDA (this gives the best model)
lda_classifier = LDA(n_components=2)
lda_x_axis = lda_classifier.fit(X, y).transform(X)
lda_classifier.score(X_test, y_test, sample_weight=None)

#Look at Decision Tree Accuracy
dt = DecisionTreeClassifier(class_weight='balanced')
dt.fit(X,y)
dt.score(X_test,y_test)

#Look at Random Forest Accuracy
rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X,y)
rf.score(X_test,y_test)

#Extra Trees Accuracy
et = ExtraTreesClassifier(class_weight='balanced')
et.fit(X,y)
et.score(X_test,y_test)

#Bagging Accuracy
bc = BaggingClassifier(dt)
bc.fit(X,y)
bc.score(X_test,y_test)

#Boosting Accuracy
ab = AdaBoostClassifier(dt)
ab.fit(X,y)
ab.score(X_test,y_test)

#Gradient Boosting Accuracy (okay model, almost as good as LDA)
gb = GradientBoostingClassifier()
gb.fit(X,y)
gb.score(X_test,y_test)

#using LDA model to predict probablilites, look at confusion matrix
#and plot ROC curve
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

#building Roc
false_positive_rate, true_positive_rate, thresholds = skrc(y_test,proba)
roc_auc = auc(false_positive_rate, true_positive_rate)
#plotting curve
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'r', label=roc_auc)
plt.legend(loc='lower right')
plt.show()
