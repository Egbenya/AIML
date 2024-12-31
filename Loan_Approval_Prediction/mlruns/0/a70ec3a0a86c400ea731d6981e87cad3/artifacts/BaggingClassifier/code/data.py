import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

# Reading data
data = pd.read_csv('data/loan_approval_dataset.csv')

# Remove spaces in column names
data.columns = data.columns.str.replace(' ','')


# drop loan_id as it is an identity seed field
data.drop('loan_id', axis=1, inplace=True)

# Define features and target variables
X = data.drop('loan_status', axis=1)
y = data['loan_status'].apply(lambda x: 1 if x == ' Approved' else 0 )

# Split the data into training, validation and test sets.
# first we split data into 2 parts, say temporary and test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

# then we split the temporary set into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=2, stratify=y_temp)

# One hot encoder for dummy variables
X_train = pd.get_dummies(X_train, dtype=int, drop_first=True)
X_val = pd.get_dummies(X_val, dtype=int, drop_first=True)
X_test = pd.get_dummies(X_test, dtype=int, drop_first=True)

# Scaling the data to prevent feature dominance.
scaler = RobustScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# print(X_train.head().to_string())
# print(X_val.head().to_string())
# print(X_test.head().to_string())
