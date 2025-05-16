# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries & load data using pandas, and preview with df.head().
2.Clean data by dropping sl_no and salary, checking for nulls and duplicates.
3.Encode categorical columns (like gender, education streams) using LabelEncoder.
4.Split features and target: X = all columns except status y = status (Placed/Not Placed)
5.Train-test split (80/20) and initialize LogisticRegression.
6.Fit the model and make predictions.
7.Evaluate model with accuracy, confusion matrix, and classification report.
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Steffy Aavlin Raj.F.S
RegisterNumber: 212224040330
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("/content/Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)
data1.isnull().sum()
data1.duplicated().sum()

le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1.head()

x = data1.iloc[:, :-1]
y = data1["status"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report1 = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report1)
```
## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

![image](https://github.com/user-attachments/assets/5b355d8a-e25c-48ce-b894-8eecbf0f3f8e)

![image](https://github.com/user-attachments/assets/dadc1af6-f0fa-4090-9a7f-180874ea9922)

![image](https://github.com/user-attachments/assets/ca4dcdad-314c-4427-8c69-6c05c739cdc7)

![image](https://github.com/user-attachments/assets/27fd5347-d1af-40d0-bdba-f1bd02ce3470)

![image](https://github.com/user-attachments/assets/f86503aa-7cd1-45b2-b7bd-8bbf950789eb)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
