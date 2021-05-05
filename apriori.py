# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from pandas_profiling import ProfileReport
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns

# %%
df = pd.read_csv('tab2.csv').rename(
    columns={
        'transid': 'transaction_id',
        'hour1': 'hour',
        'state1': 'state',
        'zip1': 'zip',
        'custAttr2': 'client_id'
    }
)
df.head()
# %%
#check the attributes
df.info()
# %%
# explore the number of fraud(1) and legit(0) transactions in the data set
df['Class'].value_counts()

# %%
# rank the states according to the number of
# transactions in descending order
df['amount'].groupby(
    df['state']).describe().sort_values(
    'count', ascending=False)

# %%
# rank the states according to the #total amount of
# transactions
df_amount = df.groupby('state')['amount'].sum(
).reset_index().sort_values('amount', ascending=False)
df_amount

# %%
aggregations = {
    'amount': 'sum'
}
total_amount_state = df.groupby('state', as_index=False).agg(aggregations)
total_amount_state.sort_values('amount', ascending=False)

total_amount_state
# %%
# plot a graph
fig_bar, ax = plt.subplots()
bars = ax.bar(
    total_amount_state['state'],
    total_amount_state['amount'],
    width=0.8, color='m')
ax.set_ylabel('Total amount of non-cash transactions')
ax.set_xlabel('State')
ax.set_title('Total amount of non-cash transactions by state')
plt.show()
# %%
# rank the states according to the number of
# fraudulent transactions in descending order
fraud_transaction_state = df['Class'].groupby(
    df['state']).sum().reset_index().sort_values(
    'Class', ascending=False)
fraud_transaction_state
# %%
aggregations = {
    'Class': 'sum'
    # 'class' : lambda x: sum(x)
}
total_fraud_transactions = df.groupby(
    'state', as_index=False).agg(aggregations)
total_fraud_transactions.sort_values('Class', ascending=False)
# %%
# plot a graph
fig_bar, ax = plt.subplots()
bars = ax.bar(
    total_fraud_transactions['state'],
    total_fraud_transactions['Class'],
    width=0.9, color='m')
ax.set_ylabel('Total number of fraud transactions')
ax.set_xlabel('State')
ax.set_title('Total number of fraud transactions by state')
plt.show()
# %%
# delete customers with only 1 transaction- 44,250 transactions left
# df_1 = df[df.groupby('client_id')['amount'].count() > 1]
df_1 = df[df.groupby('client_id')['client_id'].transform(len) > 1]
df_1
# %%
# check how many clients are unique: 14374
print(df_1['client_id'].unique())
len(df_1['client_id'].unique())
# %%
df_1.drop('state', inplace=True, axis=1)
df_1.head()
# %%
# filter out the categorical values
cat_data = df_1.select_dtypes(include=['object']).copy()
cat_data.head()
# %%
# check for null values
print(cat_data.isnull().values.sum())
# %%
# frequency distribution of categories
print(cat_data['client_id'].value_counts())
# %%
# Strip extra spaces in "id"
df_1['client_id'] = df_1['client_id'].str.strip()
# %%
# label encoding of id
lb_make = LabelEncoder()
df_1['id_code'] = lb_make.fit_transform(df_1['client_id'])
df_1[['client_id', 'id_code']]
print(df_1)
len(df_1['id_code'])
# %%
# check unique values of client_id and id_code- they should match.
len(df_1['id_code'].unique())
# %%
len(df_1['client_id'].unique())
# %%
# drop client_id column and use the encoded value- id_code
df_1.drop('client_id', inplace=True, axis=1)
df_1.head()
#%%
# divide the set into training and testing sets
train_set, test_set = train_test_split(df_1, test_size=0.50, random_state=42)
train_data = train_set
test_data = test_set
print('train set', train_data)
print('test set', test_data)
train_data
test_data
# %%
# group the dataset by clients
df_grouped = train_data.groupby(
    ['id_code', 'transaction_id'],
    as_index=True).agg(
    {'amount': 'first', 'hour': 'first', 'zip': 'first',
     'field1': 'first', 'field2': 'first', 'flag1': 'first', 'field3': 'first',
     'field4': 'first', 'indicator1': 'first', 'indicator2': 'first',
     'flag2': 'first', 'flag3': 'first', 'flag4': 'first', 'flag5': 'first',
     'Class': 'first'})
df_grouped
# %%
# group the dataset by type of transaction- legit or fraud
# 21096 legit transactions
df_legit = df_grouped[df_grouped['Class'] <= 0]
df_legit
# %%
# define one hot encoding method
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
# %%
df_legit_encoded = df_legit.applymap(encode_units)
df_legit = df_legit_encoded
df_legit
# %%
# apply apriori algorithm to the legit dataset: frequency
frq_items = apriori(df_legit, min_support=0.9, use_colnames=True)
frq_items
# %%
rules = association_rules(frq_items, metric="lift", min_threshold=1)
legit_rules = rules
legit_rules
# %%
# check how many transactions match the apriori rule from the legit #transactions dataset- 20153 out of 21096
legit_pattern = df_legit[(df_legit['zip'] == 1) & (df_legit['flag5'] == 1) & (df_legit['hour'] == 1) & (df_legit['field4'] == 1) & (df_legit['amount'] == 1)]
legit_pattern
# %%
df_fraud = df_grouped[df_grouped['Class'] >= 1]
df_fraud
# %%
df_fraud_encoded = df_fraud.applymap(encode_units)
df_fraud = df_fraud_encoded
df_fraud
# %%
freq_items = apriori(df_fraud, min_support=0.9, use_colnames=True)
freq_items
# %%
rules_f = association_rules(freq_items, metric="lift", min_threshold=1)
fraud_rules = rules_f
fraud_rules
# %%
# check how many transactions match the apriori rule from the fraud transactions dataset: 950 out of 1029
fraud_pattern = df_fraud[(df_fraud['zip'] == 1) & (df_fraud['flag5'] == 1) & (df_fraud['hour'] == 1) & (df_fraud['field4'] == 1) & (df_fraud['field1'] == 1) & (df_fraud['Class'] == 1)]
fraud_pattern