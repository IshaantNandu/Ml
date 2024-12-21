import pandas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
#Data set reading

df = pandas.read_csv("../data/data.csv")
X = df[['Weight', 'Volume']].values.reshape(-1, 2)
y = df['CO2']

# Create polynomial features
poly = PolynomialFeatures(degree=2)  # Adjust the degree as needed
X_poly = poly.fit_transform(y)

# Model training
model = LinearRegression()
model.fit(X_poly, y)

new_data = [[2300, 1300]]  # New data point with Weight and Volume
new_data_poly = poly.transform(new_data)
predictedCO2 = model.predict(new_data_poly)
print(predictedCO2)
print(r2_score(y, model.predict(X_poly)))