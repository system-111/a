# Practical 4 - Recurrent Neural Network (RNN/LSTM) for Time Series Prediction
# Dataset: Google Stock Price (Google_stock_price_train.csv / test.csv)
# Aim: Predict next day's closing stock price using LSTM

# ============================================================
# Cell 1 - Import Libraries
# ============================================================
import pandas as pd
import numpy as np

# ============================================================
# Cell 2 - Load Dataset
# ============================================================
df1 = pd.read_csv("Google_stock_price_train.csv")
df2 = pd.read_csv("Google_stock_price_test.csv")

# ============================================================
# Cell 3 - Inspect Dataset
# ============================================================
df1.info()

# ============================================================
# Cell 4 - Clean Close Column (remove commas, convert to float)
# ============================================================
df1['Close'] = df1['Close'].astype(str).str.replace(",", "").astype(float)
df2['Close'] = df2['Close'].astype(str).str.replace(",", "").astype(float)

# ============================================================
# Cell 5 - Normalize Training Data using MinMaxScaler
# ============================================================
from sklearn.preprocessing import MinMaxScaler
train_scaler = MinMaxScaler()
df1['Normalized close'] = train_scaler.fit_transform(df1['Close'].values.reshape(-1,1))

# ============================================================
# Cell 6 - Normalize Test Data using the SAME training scaler
# ============================================================
df2['Normalized close'] = train_scaler.transform(df2['Close'].values.reshape(-1,1))

# ============================================================
# Cell 7 - Prepare Sequences
# x = today's price (input), y = tomorrow's price (target)
# x shape: (-1,1,1) -> 3D required by LSTM (batch, timesteps, features)
# y shape: (-1,1)   -> 2D for Dense output
# ============================================================
x_train = df1['Normalized close'].values[:-1].reshape(-1,1,1)
y_train = df1['Normalized close'].values[1:].reshape(-1,1)

x_test  = df2['Normalized close'].values[:-1].reshape(-1,1,1)
y_test  = df2['Normalized close'].values[1:].reshape(-1,1)

# ============================================================
# Cell 8 - Build LSTM Model
# ============================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(4, input_shape=(1,1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()

# ============================================================
# Cell 9 - Train Model
# ============================================================
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=1)

# ============================================================
# Cell 10 - Evaluate on Test Data
# ============================================================
test_loss = model.evaluate(x_test, y_test)
print('Testing loss:', test_loss)

# ============================================================
# Cell 11 - Predict on Test Data
# ============================================================
pred = model.predict(x_test)

# ============================================================
# Cell 12 - Inverse Transform to Get Actual Stock Prices
# ============================================================
y_test_actual = train_scaler.inverse_transform(y_test.reshape(-1,1))
y_test_pred   = train_scaler.inverse_transform(pred.reshape(-1,1))

# ============================================================
# Cell 13 - Compare Actual vs Predicted
# ============================================================
index = 1
print("Actual:   ", y_test_actual[index])
print("Predicted:", y_test_pred[index])
