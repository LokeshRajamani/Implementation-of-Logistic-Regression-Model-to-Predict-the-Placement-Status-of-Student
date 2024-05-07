# Ex-4 :Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
# Date: 12.03.2024
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. by using google colab .
2. Upload the dataset .
3. check null , duplicated values using .isnull() and .duplicated() function respectively.
4. Import LabelEncoder and encode the dataset.
5. Import LogisticRegression from sklearn and apply the model on the dataset.
6. Predict the values of array
7. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
8. Apply new unknown values
## Program:
```
## Developed by: LOKESH R
## RegisterNumber: 212222240055

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
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
```

## Output:

![Screenshot 2024-03-12 093628](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/dc4d32bb-297b-4532-ae9e-8639734cb559)

![Screenshot 2024-03-12 093643](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/dbd0d63e-3c83-4d49-abd5-a4baa522a2f8)

![Screenshot 2024-03-12 093650](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/a27f2467-28dd-4141-b3bc-b8a463b90ea3)

![Screenshot 2024-03-12 093656](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/f80457d9-fc71-44dc-958b-66bc900b8051)

![Screenshot 2024-03-12 093713](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/27f37da4-2753-44fe-8b1c-ad3ec00aaff3)

![Screenshot 2024-03-12 093726](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/e754158a-eccf-46d6-810e-7cea6d2a42c8)


![Screenshot 2024-03-12 093733](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/d69a96d0-eb1f-48ff-8a8c-0746ccbc5103)


![Screenshot 2024-03-12 093741](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/cc9ff84b-57eb-4511-ac16-e44486a6c541)

![Screenshot 2024-03-12 093746](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/e6874dd7-aab2-491a-aa91-12c58af9f811)

![Screenshot 2024-03-12 093750](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/250cbc00-a690-4000-8a6a-e56081d8406c)

![Screenshot 2024-03-12 093756](https://github.com/LokeshRajamani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120544804/c092bdc8-c3d9-41d3-a42f-e4ed30d60960)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
