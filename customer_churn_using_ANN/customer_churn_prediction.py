#Importing Python Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Data load and preprocessing
df=pd.read_csv('Deep-Learning/customer_churn_using_ANN/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head())
df=df.drop_duplicates()
print(df)
df.drop('customerID', axis='columns',inplace=True)
print(df)
pd.to_numeric(df.TotalCharges,errors='coerce').isnull()
df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]
df.shape
df.iloc[488].TotalCharges

#Removing space from Totalcharges column
df[df.TotalCharges !=' '].shape
df=df[df.TotalCharges !=' ']
df.shape

#Changing data types
df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
print(df['MonthlyCharges'].dtype)
print(df.dtypes)

#Data visualization#

#Tenure and Number of customers
tenure_churn_no=df[df.Churn=='No'].tenure
tenure_churn_yes=df[df.Churn=='Yes'].tenure
plt.hist([tenure_churn_yes,tenure_churn_no],label=['Churn==yes','Churn==no'])
plt.xlabel('Tenure')
plt.ylabel('Number of customers')
plt.title('Customer churn prediction visualization')
plt.legend()
plt.show()

#Monthly charges and Number of customers
mc_churn_yes=df[df.Churn=='Yes'].MonthlyCharges
mc_churn_no=df[df.Churn=='No'].MonthlyCharges
plt.hist([mc_churn_yes,mc_churn_no],label=['Churn=Yes','Churn=No'])
plt.xlabel('Monthly Charges')
plt.ylabel('Number of customers')
plt.title('Customer Churn Prediction visualization')
plt.legend()
plt.show()

#Label encoding#

def print_unique_col(df):
    for column in df:
        if df[column].dtype=='object':
            print(f'{column}: {df[column].unique()}')

df.replace('No phone service','No',inplace=True)
df.replace('No internet service','No', inplace=True)


yes_no_columns=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity',
                'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                'StreamingMovies','PaperlessBilling','Churn']

for col in yes_no_columns:
    df[col].replace({"Yes":1,"No":0},inplace=True)


df['gender']=df['gender'].replace({'Female':1,'Male':0})

for col in df:
    print(f'{col}:{df[col].unique()}')

#One-hot encoding#

df_cleaned=pd.get_dummies(data=df,columns=['InternetService','Contract','PaymentMethod'])
print(df_cleaned.columns)
print(df_cleaned.dtypes)

#Scaling columns
from sklearn.preprocessing import MinMaxScaler
cols_to_scale=['tenure','MonthlyCharges','TotalCharges']
scaler=MinMaxScaler()
df_cleaned[cols_to_scale]=scaler.fit_transform(df_cleaned[cols_to_scale])
print(df_cleaned.sample(5))
for col in df_cleaned:
    print(f'{col}:{df_cleaned[col].unique()}')

#Model Building#
X=df_cleaned.drop('Churn',axis='columns')
y=df_cleaned['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=5)

import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.Dense(26,input_shape=(26,),activation='relu'),
    keras.layers.Dense(18,activation='relu'),
    keras.layers.Dense(12,activation='relu'),
    keras.layers.Dense(6,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100)
model.evaluate(X_test,y_test)

y_predict=model.predict(X_test)

y_pred=[]
for element in y_predict:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
print(y_test[:10])
print(y_pred[:10])
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,y_pred))