import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print(f"Reading in the dataset: ")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"The shape of dataset is: {df.shape}")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int)

# Splitting the data
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values

## Commenting for the model to run on validation
# del df_train['churn']
# del df_val['churn']

# Features
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

# Function to train the model
def train(df, y, C=1.0):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X, y)

    return dv, model

# Function to predict using the model
def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# Evaluating
print(f"Starting simple validation")
y_train = df_train.churn.values
y_val = df_val.churn.values

dv, model = train(df_train, y_train, C=0.5)
y_predv = predict(df_val, dv, model)

# Checking the AUC score
auc = roc_auc_score(y_val, y_predv)
print('auc = %.3f' % auc)


#Training the full model based on entire train df
y_train = df_train_full.churn.values
dv, model = train(df_train_full, y_train, C=0.5)
print(f"Training the final Model")

# Saving the model
with open('rs-churn-model.bin', 'wb') as f_out:
    pickle.dump((dv,model), f_out)

print(f"Final model saved")