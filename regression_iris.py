import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Read the data
df = pd.read_csv(' Iris.csv')

# Data preprocessing
scaler = MinMaxScaler()
df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = scaler.fit_transform(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])

# Encode the target variable
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models and parameters
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(alpha=0.01),
    'Ridge': Ridge(alpha=0.06)
}
degree = 2
poly_features = PolynomialFeatures(degree=degree)

# Training and evaluation
for name, model in models.items():
    if name == 'Linear Regression':
        X_train_model = X_train
        X_test_model = X_test
    else:
        X_train_model = poly_features.fit_transform(X_train)
        X_test_model = poly_features.transform(X_test)
    
    model.fit(X_train_model, y_train)
    y_pred = model.predict(X_test_model)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model: {name}")
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R2 value:", r2)
    print()