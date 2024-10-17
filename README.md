# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Dataset: Import the data (e.g., employee salary data) into a pandas DataFrame.
2. Handle Missing Values: Identify and either fill or remove missing values.
3. Encode Categorical Variables: Convert categorical columns (e.g., department, gender) into
numerical form using label encoding or one-hot encoding.
4. Split the Dataset: Define your features (X) and target (y), then split the data into training and testing sets.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()
datal-data.copy()
datal.head()
datal datal.drop(['s1_no', 'salary'], axis-1) datal
from sklearn.preprocessing import LabelEncoder
le-LabelEncoder()
datal ["gender"]=le.fit_transform(data1["gender"]).
data1["ssc_b"]=le.fit_transform(datal ["ssc_b"])
datal["hsc_b"]=le.fit_transform(data1["hsc_b"]) datal["hsc_s"]=le.fit_transform(datal["hsc_s"])
data1["degree_t"]=le.fit_transform(datal ["degree_t"])
datal ["workex"]=le.fit_transform(data1["workex"]) datal ["specialisation"]=le.fit_transform(datal["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
datal
x=datal.iloc[:,: -1]
y-datal.iloc[:,-1]
y
from sklearn.model selection import train test split
x_train,x_test,y_train,y_test-train_test_split(x,y,test_size=8.2, random_state=8)
from sklearn.linear_model import LogisticRegression
model LogisticRegression (solver="liblinear")
model.fit(x_train,y_train)
y_pred-model.predict(x_test)
y_pred, x_test
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
accuracy-accuracy_score(y_test,y_pred)
confusion-confusion_matrix(y_test,y_pred)
cr-classification_report(y_test,y_pred)
print("Accuracy score:", accuracy) print("\nConfusion matrix:\n", confusion)
print("\nClassification report:\n",cr)
from sklearn import metrics
cm_display-metrics.ConfusionMatrixDisplay (confusion_matrix-confusion, display_labels=[Tru
cm_display.plot()
```

## Output:
![WhatsApp Image 2024-10-17 at 10 29 01_f882bcf3](https://github.com/user-attachments/assets/e7ab9787-d2b8-4483-963b-18a7a358738d)
![WhatsApp Image 2024-10-17 at 10 29 01_4cc49825](https://github.com/user-attachments/assets/22fc3c0c-f347-4740-be86-d961119ab5ec)
![WhatsApp Image 2024-10-17 at 10 29 01_03d5ea1c](https://github.com/user-attachments/assets/84c5cc3d-04a1-4288-9af3-95710c1bb9e1)
![WhatsApp Image 2024-10-17 at 10 29 02_4488296a](https://github.com/user-attachments/assets/410852db-14cf-415b-a315-58ae905f2a08)
![WhatsApp Image 2024-10-17 at 10 29 12_82e7e1c0](https://github.com/user-attachments/assets/f1f9a8a1-63d2-4846-8a90-32b915315e3c)
![WhatsApp Image 2024-10-17 at 10 50 52_f811ca1f](https://github.com/user-attachments/assets/0c12a485-a836-48c4-8cbc-072c560cd3bd)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
