import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

#load data-set
df = pd.read_csv('train.csv')

#handle missing data
features = ['OverallQual', 'GrLivArea', 'GarageCars',  'TotalBsmtSF', 'SalePrice']
df = df[features]
df = df.dropna()

#split data into train and test sets
X = df.drop('SalePrice', axis = 1)
y = df['SalePrice']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#train random forest model
model = RandomForestRegressor(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)

#Evaluate model performance
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)


#Save the trained model
joblib.dump(model, 'house_price_model.pkl')