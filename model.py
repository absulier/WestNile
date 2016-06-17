import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import pandas as pd
from sklearn.lda import LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve as skrc
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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

#Some basic feature selection
model=DecisionTreeClassifier(class_weight='balanced')
features=[]
scores=[]
for i in X:
    features.append([i])
    model.fit_transform(X[[i]],y)
    scores.append(model.score(X_test[[i]],y_test))
    for j in X:
        features.append([i,j])
        model.fit_transform(X[[i,j]],y)
        scores.append(model.score(X_test[[i,j]],y_test))
df_f=pd.DataFrame({'features':features, 'scores':scores})
df_f=df_f.sort_values(by='scores',ascending=False)
df_f
best=[]
for i in df_f.features:
    for j in i:
        if j not in best:
            best.append(j)

X2=X[best[0:20]]
X_test2=X_test[best[0:20]]

#Building a loop to find best model and feature selection (results are lda with the 23 best features)
model=[]
score=[]
for i in range(10,len(best)):
    X2=X[best[0:i]]
    X_test2=X_test[best[0:i]]

    #running the train and test data in LDA (this typically gives the best model)
    model.append(['lda',i])
    lda= LDA(n_components=2)
    lda_x_axis = lda.fit(X2, y).transform(X2)
    score.append(lda.score(X_test2, y_test, sample_weight=None))

    #Look at Decision Tree Accuracy
    model.append(['dt',i])
    dt = DecisionTreeClassifier(class_weight='balanced')
    dt.fit(X2,y)
    score.append(dt.score(X_test2,y_test))

    #Look at Random Forest Accuracy
    model.append(['rf',i])
    rf = RandomForestClassifier(class_weight='balanced')
    rf.fit(X2,y)
    score.append(rf.score(X_test2,y_test))

    #Extra Trees Accuracy
    model.append(['et',i])
    et = ExtraTreesClassifier(class_weight='balanced')
    et.fit(X2,y)
    score.append(et.score(X_test2,y_test))

    #Bagging Accuracy
    model.append(['bc',i])
    bc = BaggingClassifier(dt)
    bc.fit(X2,y)
    score.append(bc.score(X_test2,y_test))

    #Boosting Accuracy
    model.append(['ab',i])
    ab = AdaBoostClassifier(dt)
    ab.fit(X2,y)
    score.append(ab.score(X_test2,y_test))

    #Gradient Boosting Accuracy (okay model, almost as good as LDA)
    model.append(['gb',i])
    gb = GradientBoostingClassifier()
    gb.fit(X2,y)
    score.append(gb.score(X_test2,y_test))

model_scores=pd.DataFrame({'model':model,'score':score})

for i in model_scores.index:
    if model_scores.score[i]==max(model_scores.score):
        print model_scores.score[i]
        print model_scores.model[i]

#using LDA model without feature selection to predict probablilites, look at confusion matrix
#and plot ROC curve Accuracy= .949571
X2=X[best[0:23]]
X_test2=X_test[best[0:23]]
lda= LDA(n_components=2)
lda_x_axis = lda.fit(X2, y).transform(X2)
lda.score(X_test2, y_test, sample_weight=None)

y_pred=lda.predict_proba(X_test2)
proba=pd.DataFrame(y_pred)[1]
proba.mean()

#play with the predication threshold to see falsenegative/positive trade off
y_pred2=[]
for i in proba:
    if i>.0553:
        y_pred2.append(1)
    else:
        y_pred2.append(0)

#(true negative) (false positive)
#(false negative) (true positive)
#(786)(207)
#(13)(45)
print confusion_matrix(y_test,y_pred2)

#Base line confusion matrix
#(991)(2)
#(51)(71)
y_pred=lda.predict(X_test2)
print confusion_matrix(y_test,y_pred)

#building Roc (AUC = .859)
false_positive_rate, true_positive_rate, thresholds = skrc(y_test,proba)
roc_auc = auc(false_positive_rate, true_positive_rate)
#plotting curve
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'r', label=roc_auc)
plt.legend(loc='lower right')
plt.show()
