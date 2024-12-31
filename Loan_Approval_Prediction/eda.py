from data import data


#Data Sanity check and understanding the data
print(data.head())


print(data.isnull().sum())          # checking null values

#print(data.info())                 # checking data information
print(f'There are {data.shape[0]} rows and {data.shape[1]} columns in the data')        #checking data shape

print(data.describe())              #checking data statistics

print(f"duplicates: {data.duplicated().sum()}")

print(data.nunique())

print(list(data.columns))