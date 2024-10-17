# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.
2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.
3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.
4. Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.
5. Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.
6. Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.
7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.
8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sreeviveka V.S
RegisterNumber:  2305001031
*/
```
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
x=data1.iloc[:,: -1]
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
## Output:
![image](https://github.com/user-attachments/assets/2fec06de-f31a-42ea-85d3-1cdd6f721ef2)
![Screenshot 2024-10-17 094245](https://github.com/user-attachments/assets/a92eb5ef-0536-46ec-8d6e-d2bdff4f19cd)
![Screenshot 2024-10-17 094257](https://github.com/user-attachments/assets/94f45cde-2952-43e4-aa60-bc5a2a92c61e)
![Screenshot 2024-10-17 094306](https://github.com/user-attachments/assets/b4d08aa2-816d-4a07-9559-f45f1ce55fec)
![Screenshot 2024-10-17 094345](https://github.com/user-attachments/assets/a3f3ee3a-4a68-4e90-97a3-cbc00d91c109)
![Screenshot 2024-10-17 094356](https://github.com/user-attachments/assets/729b58c6-b274-4198-96eb-5ab98699d828)
![Screenshot 2024-10-17 094412](https://github.com/user-attachments/assets/b1ab8d84-44ca-49c1-ba96-871f84511269)
![Screenshot 2024-10-17 094420](https://github.com/user-attachments/assets/4eeb5294-1e8f-4924-82ee-b0262677a8a2)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
