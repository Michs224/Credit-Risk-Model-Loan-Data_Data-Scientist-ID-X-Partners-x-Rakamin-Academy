# ## Hypothesis
# Predict which loans will default, focus on clients who will default.

import pandas as pd
import numpy as np
import os
os.listdir()



# ## The Data
# Data Source: Kaggle
# Link: https://www.kaggle.com/datasets/devanshi23/loan-data-2007-2014

loanData=pd.read_csv("loan_data_2007_2014.csv")

loanData.head()

loanData.shape

loanData.info()


# ## Target Variable
# Notify that bad or good loan, need column "loan_status"


loanData["loan_status"].unique()

# The bad ones are defined as 1, and the rest good are 0.
# Bad: 'Charged Off', 'Default',
#         'Late (31-120 days)', 'Late (16-30 days)',
#         'Does not meet the credit policy. Status:Charged Off'
# Good: Else

loanData["good_bad"]=np.where(loanData.loc[:,"loan_status"].isin(['Charged Off', 'Default',  
       'Late (31-120 days)', 'Late (16-30 days)',  
       'Does not meet the credit policy. Status:Charged Off']),1,0)

loanData["good_bad"].value_counts()

loanData[["loan_status","good_bad"]]


# pd.options.display.max_columns=None

loanData[["loan_status","good_bad"]].value_counts()






# ## Missing Values
# Due to the large number of rows and columns, it is important to check for missing values.
# So that the model is easier to predict based on these columns

missing_values=pd.DataFrame(loanData.isnull().sum()/loanData.shape[0])

missing_values=missing_values[missing_values.iloc[:,0] >0.50]
# The missing value is above 50%

missing_values.sort_values([0],ascending=False,inplace=True)
# [0] because the rows are at 0

missing_values


# It is better to delete the column with the missing value  
# because even if you try to fill it with media, mean or other techniques for missing values
# it will produce less accurate values.


loanData.drop(columns=missing_values.index,inplace=True)
# loanData.dropna(thresh=loanData.shape[0]*0.5,axis=1,inplace=True)

loanData.info()

loanData.drop("Unnamed: 0",axis=1,inplace=True)


loanData.info()


missing_values=pd.DataFrame(loanData.isnull().sum()/loanData.shape[0])
missing_values=missing_values[missing_values.iloc[:,0] >0.50]
missing_values.sort_values([0],ascending=False,inplace=True)

missing_values

# Missing values sudah terhapus





# ## Data Splitting
# Train Dataset: train a machine learning model, by learning from the patterns in this Train Data.
# Test Dataset: evaluates the machine learning model, whether when the model predicts a new data point, namely whether the test set is accurate or not.

loanData.shape

# 80% Train set dan 20% Test set

from sklearn.model_selection import train_test_split


x=loanData.drop("good_bad",axis=1)
# All columns except good_bad column target variable/responvariable (predictor).
y=loanData["good_bad"]
# Target column, which will be predicted.

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


x_train.shape,x_test.shape


y_train.shape,y_test.shape


y_train.value_counts(normalize=True)

y_test.value_counts(normalize=True)


# Imbalanced, where there are very few bad loans, we should have the same distribution, but the distribution is almost the same, so I will use the stratify=y parameter in the train_test_split function so that the y_train and y_test have a more equal distribution, because if they are not the same, then will affect our machine learning model
  
Also use the random_state=42 parameter so that when using the train_test_split function the rows will be random, that's all. So. even though we just run that, surely the results that will get random will be the same lines to make it easier to do a train test split.


x=loanData.drop("good_bad",axis=1)
y=loanData["good_bad"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

x_train.shape,x_test.shape


y_train.shape,y_test.shape

y_train.value_counts(normalize=True)


y_test.value_counts(normalize=True)


# So you can see that the distribution of y_train and y_test has become more the same.





# ## Data Cleaning
x_train.shape


# Category or non-numeric column

x_train.select_dtypes(include=["object","bool"]).columns

for col in x_train.select_dtypes(include=["object","bool"]).columns:
    print(col)
    print(x_train[col].unique())
    print()


# term : should be numeric
# emp_length : should be numeric
# issue_d, earliest_cr_line, last_pymnt_d, next_pymnt_d, last_credit_pull_d : should be datetime

col_to_clean=["term","emp_length","issue_d","earliest_cr_line",
              "last_pymnt_d","next_pymnt_d","last_credit_pull_d"]

x_train["term"].unique()

x_train["term"]=x_train["term"].str.replace(" months","")
x_train["term"]=x_train["term"].str.replace(" ","")


x_train["term"]=x_train["term"].astype(int)

x_train["term"]


x_train["emp_length"]

x_train.shape

# "Length of work in years. Possible values are between 0 and 10
# where 0 means less than one year and 10 means ten years or more."

x_train["emp_length"]=x_train["emp_length"].str.replace("+","")
x_train["emp_length"]=x_train["emp_length"].str.replace(" years","")
x_train["emp_length"]=x_train["emp_length"].str.replace("< 1 year","0")
x_train["emp_length"]=x_train["emp_length"].str.replace(" year","")

# You can fill it with a value of 0, and here I apply this.
x_train["emp_length"].fillna(0,inplace=True)

print(x_train["emp_length"])


x_train.shape

x_train["emp_length"].unique()

x_train["emp_length"]=x_train["emp_length"].astype(int)


x_train["emp_length"]

col_to_clean


col_date=['issue_d',
 'earliest_cr_line',
 'last_pymnt_d',
 'next_pymnt_d',
 'last_credit_pull_d']

x_train[col_date]

# x_train["issue_d"]=pd.to_datetime(x_train["issue_d"],format="%b-%y")

# x_train["issue_d"]

for col in col_date:
    x_train[col]=pd.to_datetime(x_train[col],format="%b-%y")

x_train[col_date]


x_test["term"]=x_test["term"].str.replace(" months","")
x_test["term"]=x_test["term"].str.replace(" ","")
x_test["term"]=x_test["term"].astype(int)

x_test["emp_length"]=x_test["emp_length"].str.replace("+","")
x_test["emp_length"]=x_test["emp_length"].str.replace(" years","")
x_test["emp_length"]=x_test["emp_length"].str.replace("< 1 year","0")
x_test["emp_length"]=x_test["emp_length"].str.replace(" year","")
x_test["emp_length"].fillna(0,inplace=True)
x_test["emp_length"]=x_test["emp_length"].astype(int)

for col in col_date:
    x_test[col]=pd.to_datetime(x_test[col],format="%b-%y")

x_test[col_to_clean].info()




# ## Feature Engineering
# Generates a new column from existing columns.

x_train.shape,y_train.shape,x_test.shape,y_test.shape

col_to_clean


x_train[col_to_clean]

x_train=x_train[col_to_clean]
x_test=x_test[col_to_clean]


del x_train["next_pymnt_d"]
del x_test["next_pymnt_d"]
# For the column next_pymnt_d, i.e. the next payment time doesn't really matter here
# even when you pay.
# So it can just be deleted, and the data used is a column other than the next_pymnt_d.

x_train.shape,x_test.shape

x_train


# To input the data into the model, it is necessary to convert it into numbers for the columns.

from datetime import date

def date_columns(df,column):
    df["months_since_"+column]=round((pd.to_datetime("2017-12-01")-df[column])/np.timedelta64(1,"M"))
    df.drop(columns=column,inplace=True)

# Apply to x_train    
date_columns(x_train,"issue_d")
date_columns(x_train,"earliest_cr_line")
date_columns(x_train,"last_pymnt_d")
date_columns(x_train,"last_credit_pull_d")

x_train

# Apply to x_test
date_columns(x_test,"issue_d")
date_columns(x_test,"earliest_cr_line")
date_columns(x_test,"last_pymnt_d")
date_columns(x_test,"last_credit_pull_d")

x_test.isnull().sum(),x_train.isnull().sum()
# It turned out that there were still missing values, so the missing values were filled in.


x_train.fillna(x_train.mean(),inplace=True)
x_test.fillna(x_train.mean(),inplace=True)

x_train.isnull().sum(),x_train.isnull().sum()



# ## Modelling
# Target:
# 1 --> (Probability Approaching 1 ) Default (Bad Loan)
# 0 --> (Probability Close to 0 ) Successfully Pay (Good Loan)

# Starting from a simple model
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()


model.fit(x_train,y_train)
y_pred=model.predict(x_test)

result=pd.DataFrame(list(zip(y_pred,y_test)),columns=["y_pred","y_test"])
result.head()


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

cm=confusion_matrix(y_test,y_pred)


sns.heatmap(cm,annot=True,fmt=".0f",cmap=plt.cm.YlGnBu)
plt.xlabel("y_pred")
plt.ylabel("y_test/y_true")
plt.show()


y_train.value_counts(normalize=True)


# Because, in this case there is an imbalanced class distribution, because there are very many 0 values and few 1 values. Therefore, we cannot use accuracy_score to measure the performance of our model in this case, because it might be misleading.

y_pred=model.predict_proba(x_test)[:,1]
y_pred

(y_pred>0.5).astype(int)

plt.hist(y_pred);


# Calculating the best threshold

from sklearn.metrics import roc_curve

fpr,tpr,thresholds=roc_curve(y_test,y_pred)

thresholds


# youden j-statistic
j=tpr-fpr
ix=np.argmax(j)
best_thresh=thresholds[ix]

best_thresh

model.predict_proba(x_test) # Class 0, class 1

y_pred=model.predict_proba(x_test)[:,1]
y_pred

y_pred=(y_pred>0.09).astype(int)

# If we predict a loan above 0.09 (9%) it is very likely that the loan will default.
# However, if it is below 0.09 (9%), it is likely that the loan will be successfully paid.


cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt=".0f",cmap=plt.cm.YlGnBu)
plt.xlabel("y_pred")
plt.ylabel("y_test/y_true")
plt.show()

model.intercept_

model.coef_

df_coeff=pd.DataFrame(model.coef_,columns=x_train.columns)
df_coeff


x_train.head()


# - term: the longer the timeframe for repaying the loan, the higher the probability that he will default.

# - emp_length: the higher, the lower the probability of default.

# - months_since_issue_d: the longer the disbursed money, the lower the probability of default.

# - months_since_earlist_cr_line: the longer it is an hour since the credit creation date, the lower the probability of default. 
#   (If the age of the credit line is old, it means that the probability of default is lower).

# - months_since_last_pymnt_d: the longer the last payment, the higher the probability that he will default.

# - months_since_last_credit_pull_d: the longer the last credit evaluation, the higher the probability that he will default. 
#   (So we have to evaluate it routinely, because if we routinely carry out evaluations, 
#   it means that the difference between the last evaluation date and the current date is getting smaller, 
#   therefore, the probability that he will fail to pay will be smaller).
