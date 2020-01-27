from sklearn import datasets
# Load CSV using Pandas from URL
import statsmodels.api as sm
import numpy as np
import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = [ 'RM', 'LSTAT', 'PTRATIO', 'MEDV']
data = pandas.read_csv(url, names=names)


# define the predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target  MEDV in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])
X = df[[“RM”, “LSTAT”,"PTRATIO"]]
y = target[“MEDV”]
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()