# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
**Date:**
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Load the dataset and perform any necessary preprocessing, such as handling missing values
 and encoding categorical variables.
 2. Initialize the logistic regression model and train it using the training data.
 3. Use the trained model to predict the placement status for the test set.
 4. Evaluate the model using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sreeviveka V.S 
RegisterNumber: 2305001031 
*/
 import pandas as pd
 data=pd.read_csv("/content/ex45Placement_Data.csv")
 data.head()
 data1=data.copy()
 data1.head()
 data1=data1.drop(['sl_no','salary'],axis=1)
 data1
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 data1["gender"]=le.fit_transform(data1["gender"])
 data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
 data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
 data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
 data1["degree_t"]=le.fit_transform(data1["degree_t"])
 data1["workex"]=le.fit_transform(data1["workex"])
 data1["specialisation"]=le.fit_transform(data1["specialisation"])
 data1["status"]=le.fit_transform(data1["status"])
 data1
x=data1.iloc[:, : -1]
 x
 y=data1.iloc[:,-1]
 y
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 from sklearn.linear_model import LogisticRegression
 model=LogisticRegression(solver="liblinear")
 model.fit(x_train,y_train)
 y_pred=model.predict(x_test)
 y_pred,x_test
 from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
 accuracy=accuracy_score(y_test,y_pred)
 confusion=confusion_matrix(y_test,y_pred)
 cr=classification_report(y_test,y_pred)
 print("Accuracy score:",accuracy)
 print("\nConfusion matrix:\n",confusion)
 print("\nClassification report:\n",cr)
 from sklearn import metrics
 cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True
 cm_display.plot()
```

## Output:
![WhatsApp Image 2024-10-17 at 10 29 01_7b9c4bb1](https://github.com/user-attachments/assets/a1e1215e-23f2-43b7-bcb4-c9d9eb9b2965)
![WhatsApp Image 2024-10-17 at 10 29 01_ab5fbf1a](https://github.com/user-attachments/assets/8ed04dd9-264e-43eb-8388-37426deb2e86)
![WhatsApp Image 2024-10-17 at 10 29 01_d25c021a](https://github.com/user-attachments/assets/79ac2eec-4633-4fcf-b61b-d759421f590c)
![WhatsApp Image 2024-10-17 at 10 29 02_e474df2c](https://github.com/user-attachments/assets/a308b347-71ad-4658-8cc1-855c20e114f3)
![WhatsApp Image 2024-10-17 at 10 29 12_c79a9f77](https://github.com/user-attachments/assets/25bc4090-43b2-44f3-95e6-91bf1f1adbdc)
![WhatsApp Image 2024-10-17 at 10 50 52_c8df71d8](https://github.com/user-attachments/assets/130d7ddd-8078-4401-8aef-a307ddb5d56c)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
