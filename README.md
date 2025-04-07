# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AHALYA S
RegisterNumber: 212223230006

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
# TOP 5 ELEMENTS
![image](https://github.com/user-attachments/assets/6dde0063-19e3-48b4-b539-b79e4340e629)

![image](https://github.com/user-attachments/assets/4f76e25e-095b-4676-b5ba-28cde9469e02)

![image](https://github.com/user-attachments/assets/ef8fcb1c-5821-47f3-9d25-dedbf97147f9)

# DATA DUPLICATE
![image](https://github.com/user-attachments/assets/8daaefa1-48ba-435b-b265-0dca868f975a)

# PRINT DATA
![image](https://github.com/user-attachments/assets/8aafe38c-478d-4716-be43-f347b84411ea)

# DATA_STATUS
![image](https://github.com/user-attachments/assets/c09432cc-bfdc-4128-a089-66b64ddf62c6)

# Y_PREDICTION ARRAY
![image](https://github.com/user-attachments/assets/e0a8a00a-6f0f-4b62-807d-90945013ca2f)

# CONFUSION ARRAY
![image](https://github.com/user-attachments/assets/0b6407e1-17f1-4d86-aa63-3ba31e1b6c96)

# ACCURACY VALUE
![image](https://github.com/user-attachments/assets/ecf89203-1b21-402b-973e-70cb8d0acd48)

# CLASSFICATION REPORT
![image](https://github.com/user-attachments/assets/86ccf83d-3f53-483f-b08e-428bbfecef32)

# PREDICTION
![image](https://github.com/user-attachments/assets/f788c7e9-ca98-41ff-9fc5-48c851ac5695)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
