# Importing the libraries
from data_preprocessing import *
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

# Part 1 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 2 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Predicting a new result
new_result=ann.predict(sc.transform([[1,140,70,41,168,30.5,0.53,25]])) > 0.5
if(new_result>0.5):
    print("This Person is diabetic")
else:
    print("This person is not diabetic")

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac_score=accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(ac_score, 2) * 100:.2f}%")