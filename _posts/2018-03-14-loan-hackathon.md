---
layout: post
title: First Hackathon
---
We just finished our first in class Hackathon to predict whether or not people are deserving of a loan, based on things like their sex, credit history, property area, education, and so on.

The biggest challenge to stand out for my group in this mini code session was that we spent most of our time cleaning our data. In fact, we ended up winning the first round because our minimum viable product was reported correctly as appose to our classmates. It was a shut out at a 2%, but it really hammered home the significance of reporting correctly the first time, and allowing a decent amount of time to make sure your work wasn't for nothing.

We mainly focused on self interpretation of the data, and RandomForest/Logistic for our models. Which scored a 76.4 and 77.78 respectively. We went column by column to account for missing values. While we were interested in imputation to fill in the gaps, but for our mvp we just brainstormed what the column was trying to say, and sometimes we made it a category by itself. Had we used imputation earlier we could have reached a 78.4 logistic score. We were also interested in using a voting classifier to create a supermodel, but alas we ran out of time. While I am satisfied with the experience a hackathon like this affords us, I plan on continuing with this challenge until I place higher then 2196/2500 on the leaderboards. The outlook is good since our logistic regression will rank 1262th.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
train = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 614 entries, 0 to 613
    Data columns (total 13 columns):
    Loan_ID              614 non-null object
    Gender               601 non-null object
    Married              611 non-null object
    Dependents           599 non-null object
    Education            614 non-null object
    Self_Employed        582 non-null object
    ApplicantIncome      614 non-null int64
    CoapplicantIncome    614 non-null float64
    LoanAmount           592 non-null float64
    Loan_Amount_Term     600 non-null float64
    Credit_History       564 non-null float64
    Property_Area        614 non-null object
    Loan_Status          614 non-null object
    dtypes: float64(4), int64(1), object(8)
    memory usage: 62.4+ KB



```python
sample = pd.read_csv('Sample_Submission_ZAuTl8O_FK3zQHh.csv')
test = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001002</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5849</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001003</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001005</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001006</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001008</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''The only difference is the target column "loan status"'''
train.shape, test.shape
```




    ((614, 13), (367, 12))




```python
'''Some column exploration. This is how we interpreted the unknown categories'''
# train.Gender.value_counts()
train.Dependents.value_counts()
# train.Education.value_counts()
# train.Married.value_counts()
# train.Property_Area.value_counts()
# train.Credit_History.value_counts()

```




    Male      489
    Female    112
    Name: Gender, dtype: int64




```python
'''
We did a little data cleaning by converting yes | no columns to 1 and 0.
We ran into some missing data for this project and we weighed a few options.
* drop the data
* self interperate (not ideal unless there are only a few simple .value_counts)
* impute the data
Unfortunately none of us knew how to impute data properly so we went with option 2.
In hindsight imputing for a small amount of the dataset is better then interpriting it.
We found on later assignments that imputing is not a good choice when it makes up over 30% of the data points.
'''
train.Gender = train.Gender.map(lambda x: 0 if x=='Female' else 1)
train.Married = train.Married.map(lambda x: 1 if x=='Yes' else 0)
train.Loan_Status = train.Loan_Status.map(lambda x: 1 if x=='Y' else 0)
train.Property_Area = train.Property_Area.map(lambda x: 0 if x == 'Urban' else 1 if x=='Semiurban' else 2)
train.Self_Employed = train.Self_Employed.map(lambda x: 0 if x=='No' else 1 if x=='Yes' else 0 if x==0 else 1)
train.Education = train.Education.map(lambda x: 0 if x=='Not Graduate' else 1 if x=='Graduate' else 0 if x==0 else 1)

```


```python
# first we remembered how to make .loc masks
# train.Dependents[train.Dependents.isnull()]=0
'''Simple code to fill Nans with 0 so they can be worked with.'''
# train.loc[train.Dependents.isnull(), 'Dependents'] = 0
train.loc[train.Dependents.isnull(), 'Dependents'] = 3
train.loc[train.Credit_History.isnull(), 'Credit_History'] = 0
train.loc[train.LoanAmount.isnull(), 'LoanAmount'] = train.LoanAmount.mean()
train.loc[train.Loan_Amount_Term.isnull(),'Loan_Amount_Term'] = train.Loan_Amount_Term.mean()
```


```python
'''Had to change the data type of a few columns'''
train.Dependents = pd.to_numeric(train.Dependents, downcast='integer' errors = 'coerce')
```




    1.0    475
    0.0    139
    Name: Credit_History, dtype: int64




```python
'''Built a simple corrilation table'''
train.corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gender</th>
      <td>1.000000</td>
      <td>0.367389</td>
      <td>0.172914</td>
      <td>-0.045364</td>
      <td>-0.023918</td>
      <td>0.058809</td>
      <td>0.082912</td>
      <td>0.107930</td>
      <td>-0.073567</td>
      <td>0.026655</td>
      <td>0.025752</td>
      <td>0.017987</td>
    </tr>
    <tr>
      <th>Married</th>
      <td>0.367389</td>
      <td>1.000000</td>
      <td>0.340684</td>
      <td>-0.017671</td>
      <td>-0.016628</td>
      <td>0.050194</td>
      <td>0.080496</td>
      <td>0.145643</td>
      <td>-0.103400</td>
      <td>-0.023626</td>
      <td>-0.000395</td>
      <td>0.084281</td>
    </tr>
    <tr>
      <th>Dependents</th>
      <td>0.172914</td>
      <td>0.340684</td>
      <td>1.000000</td>
      <td>-0.055752</td>
      <td>0.042142</td>
      <td>0.118202</td>
      <td>0.030430</td>
      <td>0.163106</td>
      <td>-0.101054</td>
      <td>-0.017523</td>
      <td>0.000244</td>
      <td>0.010118</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>-0.045364</td>
      <td>-0.017671</td>
      <td>-0.055752</td>
      <td>1.000000</td>
      <td>0.019059</td>
      <td>0.140760</td>
      <td>0.062290</td>
      <td>0.166998</td>
      <td>0.077242</td>
      <td>0.081637</td>
      <td>-0.065243</td>
      <td>0.085884</td>
    </tr>
    <tr>
      <th>Self_Employed</th>
      <td>-0.023918</td>
      <td>-0.016628</td>
      <td>0.042142</td>
      <td>0.019059</td>
      <td>1.000000</td>
      <td>0.121356</td>
      <td>0.028834</td>
      <td>0.112118</td>
      <td>-0.040244</td>
      <td>-0.001923</td>
      <td>0.022732</td>
      <td>0.005857</td>
    </tr>
    <tr>
      <th>ApplicantIncome</th>
      <td>0.058809</td>
      <td>0.050194</td>
      <td>0.118202</td>
      <td>0.140760</td>
      <td>0.121356</td>
      <td>1.000000</td>
      <td>-0.116605</td>
      <td>0.565620</td>
      <td>-0.045242</td>
      <td>0.006986</td>
      <td>0.009500</td>
      <td>-0.004710</td>
    </tr>
    <tr>
      <th>CoapplicantIncome</th>
      <td>0.082912</td>
      <td>0.080496</td>
      <td>0.030430</td>
      <td>0.062290</td>
      <td>0.028834</td>
      <td>-0.116605</td>
      <td>1.000000</td>
      <td>0.187828</td>
      <td>-0.059675</td>
      <td>-0.058795</td>
      <td>-0.010522</td>
      <td>-0.059187</td>
    </tr>
    <tr>
      <th>LoanAmount</th>
      <td>0.107930</td>
      <td>0.145643</td>
      <td>0.163106</td>
      <td>0.166998</td>
      <td>0.112118</td>
      <td>0.565620</td>
      <td>0.187828</td>
      <td>1.000000</td>
      <td>0.038801</td>
      <td>-0.034518</td>
      <td>0.044776</td>
      <td>-0.036416</td>
    </tr>
    <tr>
      <th>Loan_Amount_Term</th>
      <td>-0.073567</td>
      <td>-0.103400</td>
      <td>-0.101054</td>
      <td>0.077242</td>
      <td>-0.040244</td>
      <td>-0.045242</td>
      <td>-0.059675</td>
      <td>0.038801</td>
      <td>1.000000</td>
      <td>0.005446</td>
      <td>0.077620</td>
      <td>-0.020974</td>
    </tr>
    <tr>
      <th>Credit_History</th>
      <td>0.026655</td>
      <td>-0.023626</td>
      <td>-0.017523</td>
      <td>0.081637</td>
      <td>-0.001923</td>
      <td>0.006986</td>
      <td>-0.058795</td>
      <td>-0.034518</td>
      <td>0.005446</td>
      <td>1.000000</td>
      <td>0.018761</td>
      <td>0.432616</td>
    </tr>
    <tr>
      <th>Property_Area</th>
      <td>0.025752</td>
      <td>-0.000395</td>
      <td>0.000244</td>
      <td>-0.065243</td>
      <td>0.022732</td>
      <td>0.009500</td>
      <td>-0.010522</td>
      <td>0.044776</td>
      <td>0.077620</td>
      <td>0.018761</td>
      <td>1.000000</td>
      <td>-0.032112</td>
    </tr>
    <tr>
      <th>Loan_Status</th>
      <td>0.017987</td>
      <td>0.084281</td>
      <td>0.010118</td>
      <td>0.085884</td>
      <td>0.005857</td>
      <td>-0.004710</td>
      <td>-0.059187</td>
      <td>-0.036416</td>
      <td>-0.020974</td>
      <td>0.432616</td>
      <td>-0.032112</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC

'''Always train/test your work'''

X = train.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = train['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

rf = RandomForestClassifier(n_estimators = 100)
et = ExtraTreesClassifier(n_estimators = 100)
lr = LogisticRegression(penalty = 'l1', C=.1)
#svc = SVC(kernel = 'linear')
cross_val_score(lr,X,y)
```




    array([0.7804878 , 0.73170732, 0.79901961])




```python
'''
We tend to work hard in our exploritory data analysis and see what sticks in our initial modeling.
It should be noted at this point we were also running out of time.
So we started with a Logistic and Random Forest Classifier
'''
lr.fit(X,y)
```




    LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
rf.fit(X, y)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
'''
We decided to first make our MVP once we beat baseline.
We coded quick and dirty to convert the code for our test data.
'''
test.Gender = test.Gender.map(lambda x: 0 if x=='Female' else 1)
test.loc[test.Credit_History.isnull(), 'Credit_History'] = 0
test.Married = test.Married.map(lambda x: 1 if x=='Yes' else 0)
test.loc[test.Dependents.isnull(), 'Dependents'] = 0
test.Dependents = pd.to_numeric(test.Dependents, errors = 'coerce')
test.loc[test.Dependents.isnull(), 'Dependents']=3
test.Education = test.Education.map(lambda x: 0 if x=='Not Graduate' else 1 if x=='Graduate' else 0 if x==0 else 1)
test.Self_Employed = test.Self_Employed.map(lambda x: 0 if x=='No' else 1 if x=='Yes' else 0 if x==0 else 1)
test.loc[test.LoanAmount.isnull(), 'LoanAmount'] = test.LoanAmount.mean()
test.loc[test.Loan_Amount_Term.isnull(),'Loan_Amount_Term'] = test.Loan_Amount_Term.mean()
test.Property_Area = test.Property_Area.map(lambda x: 0 if x == 'Urban' else 1 if x=='Semiurban' else 2)
```


```python
X_test = test.drop(['Loan_ID'], axis=1)
```


```python
sample.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001015</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001022</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001031</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001035</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001051</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample.Loan_Status = lr.predict(X_test)
sample.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001022</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001031</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001035</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001051</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample['Loan_ID'] = test['Loan_ID']
sample.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001022</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001031</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001035</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001051</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample.Loan_Status.value_counts()
```




    1    279
    0     88
    Name: Loan_Status, dtype: int64




```python
sample.Loan_Status = sample.Loan_Status.map(lambda x: 'Y' if x==1 else 'N')
```


```python
sample.to_csv('l1_Predictions.csv', index = False)
```


```python
list(zip(rf.feature_importances_, X.columns))
```




    [(0.024989683569691667, 'Gender'),
     (0.025518242529179114, 'Married'),
     (0.0529070153112917, 'Dependents'),
     (0.028446945056777632, 'Education'),
     (0.023736904706004334, 'Self_Employed'),
     (0.22620565175641486, 'ApplicantIncome'),
     (0.13191627044043164, 'CoapplicantIncome'),
     (0.2103786438520365, 'LoanAmount'),
     (0.05498194603162161, 'Loan_Amount_Term'),
     (0.16923029685090094, 'Credit_History'),
     (0.05168839989565029, 'Property_Area')]




```python
probs = lr.predict_proba(X_test)
```


```python
sample_copy = sample
prob4 = probs[:,1]> .35
```


```python
sample_copy.loc[:,'Loan_Status'] = prob4
```


```python
sample_copy.loc[:,'Loan_Status'] = sample_copy.Loan_Status.map(lambda x: 'Y' if x==True else 'N')
```


```python
sample_copy.to_csv('.35 threshhold.csv', index=False)
```
