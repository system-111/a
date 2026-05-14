# Practical 1 - Linear Regression using Deep Neural Network
# Dataset: Boston Housing (boston_test.csv)
# Aim: Predict lstat (lower status population %) using DNN regression

# ============================================================
# Cell 1 - Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Cell 2 - Load Dataset
# ============================================================
BostonTrain = pd.read_csv("boston_test.csv")

# ============================================================
# Cell 3 - Preview Dataset
# ============================================================
print(BostonTrain.head())

# ============================================================
# Cell 4 - Dataset Info and Statistics
# ============================================================
BostonTrain.info()
print(BostonTrain.describe())

# ============================================================
# Cell 5 - Prepare Features and Target
# Features: all cols except ID (col 0) and lstat (last col)
# Target: lstat = % lower status population (last col in this CSV)
# ============================================================
X = BostonTrain.iloc[:,1:-1].values
Y = BostonTrain.iloc[:,-1].values

# ============================================================
# Cell 6 - Train-Test Split
# ============================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

# ============================================================
# Cell 7 - Check Shapes
# ============================================================
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# ============================================================
# Cell 8 - Import Keras
# ============================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ============================================================
# Cell 9 - Check Input Shape
# ============================================================
print("Input shape:", X_train[0].shape)

# ============================================================
# Cell 10 - Build Model
# ============================================================
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=X_train[0].shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ============================================================
# Cell 11 - Train Model
# ============================================================
model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_test, y_test))

# ============================================================
# Cell 12 - Single Prediction
# ============================================================
print("Actual Value: ", y_test[8])
sample = np.array([X_test[8]])
print("Predicted Value: ", model.predict(sample)[0][0])

# ============================================================
# Cell 13 - Evaluate Model Metrics
# ============================================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)
y_true = y_test.values if hasattr(y_test, 'values') else y_test

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)
