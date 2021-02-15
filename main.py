import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import utils
import settings

df = pd.read_csv(settings.load_path, sep='["]*;["]*', engine='python')
prediction_set = pd.read_csv(settings.store_path, sep='["]*;["]*', engine='python')

utils.trim_quotes_from_header(df)
utils.trim_quotes_from_header(prediction_set)

df.rename({'Cocunut': 'cocunut', 'Mortgage_YN': 'mortage_yn', 'AGE_AT_ORIGINATION': 'age_at_origination',
           'AGE': 'age',
           'YEARS_WITH_BANK': 'years_with_bank',
           'MARTIAL_STATUS': 'martial_status',
           'EDUCATION': 'education',
           'EMPLOYMENT': 'employment',
           'GENDER': 'gender',
           'CUST_INCOME': 'cust_income',
           'CURRENT_ADDRESS_DATE': 'current_address',
           'CURRENT_JOB_DATE': 'current_job',
           'CURRENT_WITH_BANK_DATE': 'current_with_bank_date',
           'CURRENT_BALANCE_EUR': 'current_balance_eur'}, axis=1, inplace=True)

prediction_set.rename({'Cocunut': 'cocunut',
                       'AGE': 'age',
                       'YEARS_WITH_BANK': 'years_with_bank',
                       'MARTIAL_STATUS': 'martial_status',
                       'EDUCATION': 'education',
                       'EMPLOYMENT': 'employment',
                       'GENDER': 'gender',
                       'CUST_INCOME': 'cust_income',
                       'CURRENT_ADDRESS_DATE': 'current_address',
                       'CURRENT_JOB_DATE': 'current_job',
                       'CURRENT_WITH_BANK_DATE': 'current_with_bank_date',
                       'CURRENT_BALANCE_EUR': 'current_balance_eur'}, axis=1, inplace=True)

# Prepare features
# Column 'current_with_bank_date' may remove because of 'column years_with_bank'; 'cocunut' - removed
# Potential customers cvs doesn't have column 'age_at_origination' =>  drop
df.drop('cocunut', axis=1, inplace=True)
df.drop('current_with_bank_date', axis=1, inplace=True)
df.drop('age_at_origination', axis=1, inplace=True)
prediction_set.drop('current_with_bank_date', axis=1, inplace=True)
prediction_set.drop('cocunut', axis=1, inplace=True)

# Analyse of distribution of categorical data
utils.plotting_data_for_analyse_distribution_of_cardinal(df, prediction_set)

# After analyse of martial_status => removing '*noval*' data (43)
indexNames = df[df['martial_status'] == '*noval*'].index
# Delete these row indexes from dataFrame
df.drop(indexNames, axis=0, inplace=True)

indexNames = prediction_set[prediction_set['martial_status'] == '*noval*'].index
# Delete these row indexes from dataFrame
prediction_set.drop(indexNames, axis=0, inplace=True)

# Preprocessing data
# Encoding Categorical values
df['mortage_yn'] = df['mortage_yn'].map({'N': 0, 'Y': 1}).astype(np.int)
df['gender'] = df['gender'].map({'F': 0, 'M': 1}).astype(np.int)
df['martial_status'] = df['martial_status'].map({'W': 0, 'D': 1, 'S': 2, 'M': 3}).astype(np.int)
df['education'] = df['education'].map({'OTH': 0, 'PRI': 1, 'PHD': 2, 'MAS': 3, 'SEC': 4, 'PRS': 5, 'BCR': 6, 'HGH': 7}).astype(np.int)
df['employment'] = df['employment'].map({'OTH': 0, 'PRI': 1, 'SFE': 2, 'RET': 3, 'STE': 4, 'PVE': 5}).astype(np.int)
prediction_set['gender'] = prediction_set['gender'].map({'F': 0, 'M': 1}).astype(np.int)
prediction_set['martial_status'] = prediction_set['martial_status'].map({'W': 0, 'D': 1, 'S': 2, 'M': 3}).astype(np.int)
prediction_set['education'] = prediction_set['education'].map({'OTH': 0, 'PRI': 1, 'PHD': 2, 'MAS': 3, 'SEC': 4, 'PRS': 5, 'BCR': 6, 'HGH': 7}).astype(np.int)
prediction_set['employment'] = prediction_set['employment'].map({'TEA': 0, 'SFE': 1, 'OTH': 2, 'RET': 3, 'STE': 4, 'PVE': 5}).astype(np.int)

# cust_income and current_balance_eur converted to float
df['cust_income'] = df['cust_income'].str.replace(',', '.').astype(np.float64)
df['current_balance_eur'] = df['current_balance_eur'].str.replace(',', '.').astype(np.float64)
prediction_set['cust_income'] = prediction_set['cust_income'].str.replace(',', '.').astype(np.float64)
prediction_set['current_balance_eur'] = prediction_set['current_balance_eur'].str.replace(',', '.').astype(np.float64)

# Cleaning data
# The timestamp limitations in pandas is only for 584 years (1677-2262)
df['current_address'] = df['current_address'].str.replace('9999-10-01', '2262-04-11')
df['current_job'] = df['current_job'].str.replace('9999-10-01', '2262-04-11')
prediction_set['current_address'] = prediction_set['current_address'].str.replace('9999-10-01', '2262-04-11')
prediction_set['current_job'] = prediction_set['current_job'].str.replace('9999-10-01', '2262-04-11')

df['current_address'] = pd.to_datetime(df['current_address']).astype(np.int)//10 ** 9
df['current_job'] = pd.to_datetime(df['current_job']).astype((np.int))//10 ** 9
prediction_set['current_address'] = pd.to_datetime(prediction_set['current_address']).astype((np.int))/10 ** 9
prediction_set['current_job'] = pd.to_datetime(prediction_set['current_job']).astype((np.int))//10 ** 9


# Distributtion of numeric
utils.plotting_data_analyse_distribution_of_numeric(df)

print(df.head(10))
print(df.info())
print("****************************************************")
print(prediction_set.head(10))
print(prediction_set.info())

# Setting Features and y
X = np.asarray(df[['age', 'years_with_bank', 'martial_status', 'education', 'employment', 'gender', 'cust_income',
                  'current_address', 'current_job', 'current_balance_eur']])
y = np.asarray(df['mortage_yn'])

# Normalization of dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Splitting in two datasets: training and test
from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=17)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_cv.shape,  y_cv.shape)

# Modeling Logistic Regression with regularization
LR = LogisticRegression(C=0.09, solver='liblinear').fit(X_train, y_train)
y_pred_cv = LR.predict(X_cv)
y_pred_cv_prob = LR.predict_proba(X_cv)

# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_cv, y_pred_cv)
utils.visualize_confusion_matrix(cnf_matrix)

# Confusion Matrix Evaluation Metrics
print("Accuracy:", metrics.accuracy_score(y_cv, y_pred_cv))
print("Precision:", metrics.precision_score(y_cv, y_pred_cv))
print("Recall:", metrics.recall_score(y_cv, y_pred_cv))

# Modeling Potential customer
y_test = utils.potential_customers(prediction_set, LR)
utils.preprocessing_data_for_writing(prediction_set, y_test)

utils.plotting_data_after_training_potential_customers()


