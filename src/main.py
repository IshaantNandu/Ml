#Data set reading
import pandas
df = pandas.read_csv("../data/data.csv")
X = df[['Weight', 'Volume']].values
X = X.reshape(-1, 2)
y = df['CO2']

from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[2300, 1300]])
print(predictedCO2)