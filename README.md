# CODSOFT_3

 Task 3: Customer Churn Prediction

 🔍 Problem Statement
Predict whether a customer will churn from a subscription-based service using their demographic and behavioral data.

 📌 Dataset Features
- **CreditScore**, **Geography**, **Gender**, **Age**, **Tenure**
- **Balance**, **NumOfProducts**, **HasCrCard**, **IsActiveMember**, **EstimatedSalary**
- **Target:** `Exited` (1 = churned, 0 = retained)

🔹 **Step 1: Import Libraries**

Import necessary Python libraries:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

 🔹 **Step 3: Load & Explore the Dataset**

```python
df = pd.read_csv('Churn.csv')
df.head()
df.info()
df.describe()
```

🔹 **Step 4: Data Cleaning & Preprocessing**

* Drop irrelevant columns: `'RowNumber'`, `'CustomerId'`, `'Surname'`
* Encode categorical features:

  * `Gender`: Label Encoding (Male = 1, Female = 0)
  * `Geography`: One-hot encoding
* Separate target (`Exited`) and features
* Split data into train-test sets (80/20)
* Scale numerical features

```python
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

 🔹 **Step 5: Model Training**

Train the following models:

 ✅ Logistic Regression

```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
```

 ✅ Random Forest

```python
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
```

 ✅ XGBoost Classifier

```python
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
```

 🔹 **Step 6: Model Evaluation**

Evaluate using accuracy, precision, recall, F1-score.

```python
y_pred_xgb = xgb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
```

🔹 **Step 7: Model Comparison**

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 81.1%    |
| Random Forest       | 86.75%   |
| XGBoost             | 86.95% ✅ |

 🔹 **Step 8: Final Insight**

* XGBoost performed best in identifying churn-prone customers.
* Key predictors: `Age`, `Balance`, `IsActiveMember`, and `NumOfProducts`.




 

