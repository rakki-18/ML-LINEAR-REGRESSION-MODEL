from sklearn import datasets
# Loading CSV using Pandas from URL
import statsmodels.api as sm
import numpy as np
import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# for the coloumn names
names = [ 'RM', 'LSTAT', 'PTRATIO', 'MEDV']
data = pandas.read_csv(url, names=names)


# define the predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target  MEDV in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])
#these coloumns are taken to be given input
X = df[[“RM”, “LSTAT”,"PTRATIO"]]
# MEDV is set as the target coloumn to be predicted
y = target[“MEDV”]
# trying to find the best fit line
model = sm.OLS(y, X).fit()
# making predictions based on the input
predictions = model.predict(X)
# gives the summary of the dataset predicted
model.summary()
