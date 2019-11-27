#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#we use the best performance code in Competition1 as reference

#Importing libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn import preprocessing#z_score to remove outliers


#Importing training and testing datasets
train = pd.read_csv("tcd-ml-1920-group-income-train.csv")
test = pd.read_csv("tcd-ml-1920-group-income-test.csv")

#DATA PROCESSING STEPS
rename_cols = {"Total Yearly Income [EUR]":'Income'}
train = train.rename(columns=rename_cols)
test = test.rename(columns=rename_cols)
yy = train["Income"]
x_len=train.shape[0]

#Removing outliers
"""
train["Income1"]=yy
train["Income"] = preprocessing.scale(train["Income"])
train = train.drop(train[(train.Income > 3)].index)
print(train.head())
y = train["Income1"]
train=train.drop('Income1', axis=1)
train=train.drop('Income', axis=1)
print(train.head())
print(x_len)#1025071
"""

data = pd.concat([train,test],ignore_index=True)
data=data.drop('Income', axis=1)

#Changing resonable valus
data['Gender'].replace('0', 'unknown', inplace=True)
data['Gender'].replace('f', 'female', inplace=True)

data["University Degree"].replace('0', 'No', inplace=True)

data["Housing Situation"].replace('0', 'unknown', inplace=True)
data["Housing Situation"].replace('nA', 'unknown', inplace=True)

 ##change the years type to be float
data['Work Experience in Current Job [years]'].replace('#NUM!', '0', inplace=True)
work_ex = [float(x) for x in data['Work Experience in Current Job [years]']]
data['Work Experience in Current Job [years]'] = work_ex

 ##change the income type to be float
extra = [x.replace(' EUR', '') for x in data['Yearly Income in addition to Salary (e.g. Rental Income)']]
numerical_extra = [float(x) for x in extra]
data['Yearly Income in addition to Salary (e.g. Rental Income)'] = numerical_extra

data['Country'].replace('0', 'unknown', inplace=True)


#Processing the null-values data
rename_cols = {"Year of Record":'Year'}
data = data.rename(columns=rename_cols)
aa=data.Age.median()
bb=data.Year.median()
#print(aa,bb)

#Replacing numerical data with the median, categorical data with 'unknown'
fill_col_dict = {
 'Age': 35.0,
 'Year': 1979.0,
 'Gender':'unknown',
 'Profession': 'unknown',
 'University Degree': 'unknown',
 'Country': 'unknown',
 'Satisfation with employer':'unknown',
}
for col in fill_col_dict.keys():
    data[col] = data[col].fillna(fill_col_dict[col])
data.info()

#Adding more combining features for the dataset
def create_cat_con(df,cats,cons,normalize=True):
    for i,cat in enumerate(cats):
        vc = df[cat].value_counts(dropna=False, normalize=normalize).to_dict()
        nm = cat + '_FE_FULL'
        df[nm] = df[cat].map(vc)
        df[nm] = df[nm].astype('float32')
        for j,con in enumerate(cons):
#             print("cat %s con %s"%(cat,con))
            new_col = cat +'_'+ con
            print('timeblock frequency encoding:', new_col)
            df[new_col] = df[cat].astype(str)+'_'+df[con].astype(str)
            temp_df = df[new_col]
            fq_encode = temp_df.value_counts(normalize=True).to_dict()
            df[new_col] = df[new_col].map(fq_encode)
            df[new_col] = df[new_col]/df[cat+'_FE_FULL']
    return df

cats = ['Year','Hair Color','Wears Glasses', 'Gender', 'Country',
        'Profession', 'University Degree','Age','Housing Situation',
       'Satisfation with employer','Yearly Income in addition to Salary (e.g. Rental Income)']
cons = ['Size of City','Body Height [cm]',
        'Crime Level in the City of Employement','Work Experience in Current Job [years]']

data = create_cat_con(data,cats,cons)
#labelEncoder
for col in train.dtypes[train.dtypes == 'object'].index.tolist():
    feat_le = LabelEncoder()
    feat_le.fit(data[col].unique().astype(str))
    data[col] = feat_le.transform(data[col].astype(str))

del_col = set(['Instance'])
features_col =  list(set(data) - del_col)

#import the model
from sklearn.model_selection import train_test_split
X_train,X_test = data[features_col].iloc[:x_len],data[features_col].iloc[x_len:]#1048574
Y_train = yy
x_train,x_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state=1234)
X_train.shape

#model paramters
params = {
          'max_depth': 20,
          'learning_rate': 0.001,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
         }
trn_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_val, label=y_val)
clf = lgb.train(params, trn_data, 50000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
pre_test_lgb = clf.predict(X_test)


#Evaluate the model performance with MAE score
from sklearn.metrics import mean_absolute_error
pre_val_lgb = clf.predict(x_val)
mae_predict = mean_absolute_error(y_val,pre_val_lgb)
print(mae_predict)

#Output the predicted data
sub_df = pd.DataFrame({'Income':pre_test_lgb})
sub_df.head()
sub_df.to_csv("final.csv",index=False)