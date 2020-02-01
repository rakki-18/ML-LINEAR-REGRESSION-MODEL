import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#inputting the dataset
df = pd.read_csv(r'C:\Users\Rakki\Downloads\Iris.csv')
printf(df.head(5))
# we are importing many libraries from sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
#we are inputting every coloumn except species 
x = df.drop('Species', axis=1)
# our target is to predict the species
y = df['Species']
#training and testing the model using imports available in sklearn
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
# we are using logistic regression import from sklearn
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
# predicting if the species output comes out to be 0 or 1
predictions = logmodel.predict(x_test)
# we are printing the predicted datasets along with the actual dataset
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
# we are checking the accuracy of our predictions 
print(accuracy_score(y_test, predictions))
