# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:50:55 2017

@author: s1883483
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:47:47 2017

@author: s1883483
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from matplotlib import style
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

#Read data, skip the first row and read header

Raw_Data= pd.read_csv("C:/Users/s1883483/Desktop/selfstudy/Loan case study/Raw_Data/LoanStats_2017Q2.csv", skiprows=[0, 105456, 105455] , header=0)

# Replace null id with row index, from 0 to 105450

Raw_Data["id"].fillna(Raw_Data.index.to_series(), inplace=True)

# Data Exploration

Raw_Data.head(5)

Raw_Data.info()

pd.DataFrame.describe(Raw_Data)

# evaluate the Null_Ratio list, drop the high Null_Ratio columns(48%) and keep joint accounts related columns in case we need.

Null_Ratio = Raw_Data.isnull().sum(axis=0).sort_values(ascending=False)/len(Raw_Data)

Null_Columns = Null_Ratio[Null_Ratio>0.4].index.tolist()

# find columns has skewed data (one group larger than 95000). columns excluded.

Data_Exploration={}

Columns_List = list(Null_Ratio[Null_Ratio<0.4].index)

for column in Columns_List:
    Data_Exploration[column]=(Raw_Data.groupby(column).size().sort_values(ascending=False).head(5).values)

Skewed_Columns=[]

for column in Data_Exploration:
    if Data_Exploration[column][0]>95000:
        Skewed_Columns.append(column)

Drop_Columns = Null_Columns + Skewed_Columns

Kept_Columns =[column for column in Raw_Data.columns.values.tolist() if column not in  Drop_Columns]

# Review all the variables in Kept_Columns from dictionary. Keep important variables.
# explore the difference between funded_amnt & loan_amnt. result is same. so we keep funded_amnt.

Raw_Data["Diff"]=Raw_Data["funded_amnt"] - Raw_Data["loan_amnt"]

Raw_Data["Diff"].describe()

# Variables exploration, segment all variables into five segmentations(Loan, Income, Credit, Balance, Geograhpy).

Loan_Variables = ["loan_amnt",  "term", "installment"]

Customer_Variables = ['id', "emp_length", "home_ownership", "int_rate", "loan_status", "purpose", "application_type", "total_cu_tl"]

Income_Variables =["annual_inc", "annual_inc_joint","dti","dti_joint" ]

Credit_Variables = ["delinq_2yrs", "inq_last_6mths", "pub_rec", "grade"]

Balance_Variables = ["tot_cur_bal", "revol_bal", "revol_util", "out_prncp", "total_pymnt", "last_pymnt_amnt","il_util", "bc_util", "revol_bal_joint"]

Geo_Variables = ["zip_code", "addr_state"]

# Geo_Variables are not points of interest right not.

Raw_Data_Selected = pd.DataFrame() 

Variables = Loan_Variables + Customer_Variables + Income_Variables + Credit_Variables + Balance_Variables

Raw_Data_Selected = Raw_Data[Variables]

# Label data for supervised learning (loan_status)
# Visualization - loan_status
fig,ax=plt.subplots()

style.use('ggplot')
Status_Type = Raw_Data_Selected.groupby("loan_status").size().sort_values(ascending=False).index.tolist()
Status_Count = Raw_Data_Selected.groupby("loan_status").size().sort_values(ascending=False).tolist()
ax.bar(range(len(Status_Type)), Status_Count)
plt.xticks(range(len(Status_Type)), Status_Type, fontsize=5)
plt.show()
ax.set_xlabel("Loan Status")
ax.set_ylabel("Count")
fig.savefig('loan_status.png', dpi=1200)

# There are in totall 7 status(). To simplify this case, we only include two status, Fully Paid and  Default. Drop 'Current', and label late payments as default. (0 = not delaied cases, 1 = default) 

Raw_Data_Selected = Raw_Data_Selected[Raw_Data_Selected['loan_status']!='Current']
Raw_Data_Selected['loan_status_fixed'] = Raw_Data_Selected['loan_status'].map({'Fully Paid':0,'Charged Off':1,'Default':1,'In Grace Period':1,'Late (16-30 days)':1,'Late (31-120 days)':1})

# visualization - loan_status_fixed. Default rate = 34.3%

Default_Rate = Raw_Data_Selected.groupby(['loan_status_fixed']).size()

fig,ax=plt.subplots()

Status_Type = Raw_Data_Selected.groupby("loan_status_fixed").size().sort_values(ascending=False).index.tolist()
Status_Count = Raw_Data_Selected.groupby("loan_status_fixed").size().sort_values(ascending=False).tolist()
ax.bar(range(len(Status_Type)), Status_Count)
plt.xticks(range(len(Status_Type)), Status_Type, fontsize=5)
plt.show()
ax.set_xlabel("Loan Status Fixed")
ax.set_ylabel("Count")
fig.savefig('loan_status_fixed.png', dpi=1200)

# Evaluate the relation between 'application_type' and 'loan_status_fixed'. We find out that, Joint application has only 10% higher default rate. Drop the 'application_type' after we have our joint accounts value fixed. Combine annual_inc, annual_inc_joint, dti, dti_joint, revol_bal, revol_bal_joint, application_type. Encapsulate account holder's income and dti information into annual_inc_fixed and dti_fixed 

Raw_Data_Selected.groupby(['application_type', 'loan_status_fixed']).size()

Raw_Data_Selected['annual_inc_fixed'] = (Raw_Data_Selected['application_type']=='Individual')*Raw_Data_Selected['annual_inc'].fillna(0) + (Raw_Data_Selected['application_type']=='Joint App')*Raw_Data_Selected['annual_inc_joint'].fillna(0)

Raw_Data_Selected['dti_fixed'] = (Raw_Data_Selected['application_type']=='Individual')*Raw_Data_Selected['dti'].fillna(0) + (Raw_Data_Selected['application_type']=='Joint App')*Raw_Data_Selected['dti_joint'].fillna(0)

Raw_Data_Selected['revol_bal_fixed'] = (Raw_Data_Selected['application_type']=='Individual')*Raw_Data_Selected['revol_bal'].fillna(0) + (Raw_Data_Selected['application_type']=='Joint App')*Raw_Data_Selected['revol_bal_joint'].fillna(0)

# Convert term, int_rate, revol_util into proper format

Raw_Data_Selected['term'] = Raw_Data_Selected['term'].str.split(' ').str[1]

Raw_Data_Selected['int_rate'] = Raw_Data_Selected['int_rate'].str.split('%').str[0]
Raw_Data_Selected['int_rate'] = Raw_Data_Selected.int_rate.astype(float)/100

Raw_Data_Selected['revol_util'] = Raw_Data_Selected['revol_util'].str.split('%').str[0]
Raw_Data_Selected['revol_util'] = Raw_Data_Selected.revol_util.astype(float)/100

# Quantify ordinal data into numeric value. Map emp_length and grade.

Raw_Data_Selected.groupby('emp_length').size()
Raw_Data_Selected['emp_length_fixed'] = Raw_Data_Selected['emp_length'].map({'1 year':1,'10+ years':10,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6, '7 years':7, '8 years':8, '9 years':9, '< 1 year':0.5})
Raw_Data_Selected.fillna(0, inplace=True)

Raw_Data_Selected.groupby('grade').size()
Raw_Data_Selected['grade_fixed'] = Raw_Data_Selected['grade'].map({'A':7,'B':6,'C':5,'D':4,'E':3,'F':2,'G':1})

# Default rates for Customer with different home ownerships. Map categoritical data - 'home_ownership'. Generate extra columns to categorize "Mortgage", 'OWN', and 'RENT' home ownership status. use numeric value 1 to stand for positive. There is only 1 case for 'ANY', so ignore this type.

Raw_Data_Selected.groupby(['home_ownership', 'loan_status_fixed']).size()

Raw_Data_Selected['MORTGAGE'] = 1*(Raw_Data_Selected['home_ownership']=='MORTGAGE').fillna(0)
Raw_Data_Selected['OWN'] = 1*(Raw_Data_Selected['home_ownership']=='OWN').fillna(0)
Raw_Data_Selected['RENT'] = 1*(Raw_Data_Selected['home_ownership']=='RENT').fillna(0)

# Evaluate  'purpose'. To avoid including several extra columns,  'purpose' column dropped. 

Raw_Data_Selected.groupby(['purpose', 'loan_status_fixed']).size()

# Drop not necessary columns. Get the new data frame. Check null value portion again. Drop column 'il_util' which has  Null value. fillna use mean for 'bc_util' and 'revol_util' which has low portion of Null values respectively. 

Raw_Data_Final = Raw_Data_Selected.drop(['id', 'home_ownership', 'emp_length', 'loan_status', 'purpose', 'application_type', 'annual_inc', 'annual_inc_joint', 'dti', 'dti_joint', 'grade', 'revol_bal', 'revol_bal_joint', 'il_util'], 1)

Null_Ratio_Final = Raw_Data_Final.isnull().sum(axis=0).sort_values(ascending=False)/len(Raw_Data_Final)

Raw_Data_Final['bc_util'].fillna(Raw_Data_Final['bc_util'].median(), inplace=True)
Raw_Data_Final['revol_util'].fillna(Raw_Data_Final['revol_util'].mean(), inplace=True)

# multicollinearity and correlation coefficient check. Draft correlation matrix using Searborn. Avoid the usage of both variables that have high correlation. 

corr = Raw_Data_Final.corr()
plt.figure(figsize=(16, 16))
corr_map = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
figure = corr_map.get_figure()    
figure.savefig('C:/Users/s1883483/Desktop/selfstudy/Loan case study/Logistic_Regression/corr_conf.png', dpi=1000)

# We find out 'int_rate' is highly negative related with 'grade_fixed'. The higher 'grade_fixed' means better credit which also means lower 'int_rate'. We drop 'int_rate' as grade encapsulates more information.
# We find out "last_pymnt_amnt" and "total_pymnt" has 0.93 correlation.
# We also find out "loan_amnt" and "installment" are highly correlated which makes sense. We select 'installment' as it combines the information of both 'term' and 'loan_amnt'. 
# 'bc_util' is highly related with 'revol_util'. We use 'bc_util' to represent the balance part. And 'tot_cur_bal' is related with 'revol_bal_fixed', we keep 'revol_bal_fixed' as we already have total balance related information.
# 'dti_fixed' should be calculated based on 'installment' and 'annual_inc_fixed'. So we keep 'annual_inc_fixed' only to aviod duplicate information.
# Keep 'pub_rec', drop 'delinq_2yrs' and 'inq_last_6mths' as they hardly make sense. 

Raw_Data_Selected.groupby(['delinq_2yrs']).size()
Raw_Data_Selected.groupby(['inq_last_6mths']).size()
Raw_Data_Selected.groupby(['pub_rec']).size()

# Model variables. And labeled data Y.

Y = Raw_Data_Final['loan_status_fixed']

Model_Variables = Raw_Data_Final.drop([ 'loan_status_fixed'], 1)
# drop waiting list 'last_pymnt_amnt', 'loan_amnt', 'revol_util', 'tot_cur_bal', 'annual_inc_fixed', 'inq_last_6mths', 'term',


Headers = Model_Variables.columns.values.tolist()
Headers.remove('loan_status_fixed')

# Preprocessing use feature scaling

Model_Variables = pd.DataFrame(preprocessing.scale(Model_Variables), columns=Headers)

# Divide training set and testing set, use stratify keep the class portion in testing set as well

X_train, X_test, y_train, y_test = train_test_split(Model_Variables, Y, stratify=Y, test_size=0.3,  random_state=0)

# Logistic regression. C in sklearn is the regulization metrics in cost function. It's a inverse to  regulization strength. So if C is too small, it will cause high bias(underfit). In opposite, it will cause overfit. C will be determined by crossvalidation later after we determine our final logistic model.  We set class_weight ='balanced' option to set weight for the false prediction in the loss function.

lr = LogisticRegression(C=1, class_weight = 'balanced')

# Run logistic regression on each individual variable. Predit on test set to see accuracy. We find that the Accuracy_Results is showing almost 96% of accuracy. This is because the default rate is only 3.8%, similar to anomaly detection. Use F1 score instead. Increase the recall contribution in F1 score as it's better to predict all the defaults rather than miss someone.

Scores = pd.DataFrame(Headers, columns=['variables'])

Accuracy_Results = {}
F1_Results={}
Recall_Results={}
Precision_Results={}

for column in Headers: 
    lr.fit(X_train.as_matrix([column]), y_train)
    test = lr.predict(X_test.as_matrix([column]))
    Accuracy_Results[column] =accuracy_score(y_test, test)
    F1_Results[column]=f1_score(y_test, test)
    Recall_Results[column] = recall_score(y_test, test)
    Precision_Results[column] = precision_score(y_test, test)
    
Scores['accuracy'] = Accuracy_Results.values()
Scores['f1'] = F1_Results.values()
Scores['recall'] = Recall_Results.values()
Scores['precision'] = Precision_Results.values()

# Review variables with the correlation matrix we have above. Drop the variables which have high correlation with others and low F1 score. 'out_prncp', 'last_pymnt_amnt', 'total_pymnt' has 95%+ accuracy because we drop all 'current' status customers. So who fully paid customer for sure to have all their loan paid and with high 'total_pymnt'. 'last_pymnt_amnt' is a good indicator.

# 'home_ownership' evaluation. coef list [ 0.1252331 ,  0.14447062,  0.37599663] coef is indicating 'RENT' is taking a higher portion when predicting default.
    
lr.fit(X_train.as_matrix(['MORTGAGE', 'OWN', 'RENT']), y_train)
test = lr.predict(X_test.as_matrix(['MORTGAGE', 'OWN', 'RENT']))
Scores.loc[len(Scores)] = ['home_ownership', accuracy_score(y_test, test), f1_score(y_test, test), recall_score(y_test, test),  precision_score(y_test, test)]
print (lr.coef_)

# credit_combined parameter evaluation. coef list [ 0.13334161 -0.01651481  0.00476109 -0.34423472  0.12192683] is indicating 'delinq_2yrs', 'int_rate' contribute some portion, and 'grade_fixed' contributes majority. As 'int_rate' is highly related with 'grade_fixed'. we keep 'grade_fixed'.

lr.fit(X_train.as_matrix(['delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'grade_fixed', 'int_rate']), y_train)
test = lr.predict(X_test.as_matrix(['delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'grade_fixed', 'int_rate']))
Scores.loc[len(Scores)] = ['credit_combined', accuracy_score(y_test, test), f1_score(y_test, test), recall_score(y_test, test),  precision_score(y_test, test)]
print (lr.coef_)

# balance_combined parameter evaluation. coef list [ 0.05861168  0.29084989  0.13267624 -0.23961512] indicates 'revol_util' and 'tot_cur_bal' are important factors. The higher revolving line utilization rate is an indicator for default. and total balance is opposite.
lr.fit(X_train.as_matrix(['revol_bal_fixed', 'revol_util', 'bc_util', 'tot_cur_bal']), y_train)
test = lr.predict(X_test.as_matrix(['revol_bal_fixed', 'revol_util', 'bc_util', 'tot_cur_bal']))
Scores.loc[len(Scores)] = ['balance_combined', accuracy_score(y_test, test), f1_score(y_test, test), recall_score(y_test, test),  precision_score(y_test, test)]
print (lr.coef_)

# income_loan_combined parameter evalution.coef list [ 0.2234889   0.4603718  -0.16042673 -0.11032809] is indicating dti_fixed', 'installment', 'last_pymnt_amnt' are important factors. however 'last_pymnt_amnt' and 'installment' are correlated. pick 'last_pymnt_amnt' as it's performing much better as an individual.

lr.fit(X_train.as_matrix(['dti_fixed', 'loan_amnt', 'annual_inc_fixed', 'last_pymnt_amnt', 'installment']), y_train)
test = lr.predict(X_test.as_matrix(['dti_fixed',  'loan_amnt', 'annual_inc_fixed', 'last_pymnt_amnt', 'installment']))
Scores.loc[len(Scores)] = ['income_loan_combined', accuracy_score(y_test, test), f1_score(y_test, test), recall_score(y_test, test),  precision_score(y_test, test)]
print (lr.coef_)

# customer_combined parameter evaluation. coef list [-0.14775042  0.02072334 -0.45907533] indicating emp_length_fixed is another indicator like grade.
lr.fit(X_train.as_matrix(['emp_length_fixed', 'annual_inc_fixed', 'grade_fixed']), y_train)
test = lr.predict(X_test.as_matrix(['emp_length_fixed', 'annual_inc_fixed', 'grade_fixed']))
Scores.loc[len(Scores)] = ['customer_combined', accuracy_score(y_test, test), f1_score(y_test, test), recall_score(y_test, test),  precision_score(y_test, test)]
print (lr.coef_)

# all_valurable_combined parameter evaluation.
lr.fit(X_train.as_matrix(['MORTGAGE', 'OWN', 'RENT', 'grade_fixed', 'revol_util', 'tot_cur_bal',  'emp_length_fixed' ]), y_train)
test = lr.predict(X_test.as_matrix(['MORTGAGE', 'OWN', 'RENT', 'grade_fixed', 'revol_util', 'tot_cur_bal', 'emp_length_fixed' ]))
Scores.loc[len(Scores)] = ['all_valurable_combined', accuracy_score(y_test, test), f1_score(y_test, test), recall_score(y_test, test),  precision_score(y_test, test)]
print (lr.coef_)

# Support vector machine solution. the score displayed is very close to logistic regression with same input variables
svm = SVC(class_weight='balanced')
svm.fit(X_train.as_matrix(['MORTGAGE', 'OWN', 'RENT', 'grade_fixed', 'revol_util', 'tot_cur_bal',  'emp_length_fixed' ]), y_train)
test = svm.predict(X_test.as_matrix(['MORTGAGE', 'OWN', 'RENT', 'grade_fixed', 'revol_util', 'tot_cur_bal', 'emp_length_fixed' ]))
Scores.loc[len(Scores)] = ['svm_all_valurable_combined', accuracy_score(y_test, test), f1_score(y_test, test), recall_score(y_test, test),  precision_score(y_test, test)]
