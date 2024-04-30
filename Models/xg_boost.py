# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dataset/diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the XG Boost  model on the Training set
import xgboost
classifier = xgboost.XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # [Predicted_Result,Actual_Result]

# Predicting a new result
new_result=classifier.predict(sc.transform([[1,140,70,41,168,30.5,0.53,25]]))

if(new_result==1):
    print("This Person is diabetic")
else:
    print("This person is not diabetic")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score ,precision_score, recall_score, f1_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac_score=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)

print(cm)
print("Accuracy: {:.2f}%".format(ac_score * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))




