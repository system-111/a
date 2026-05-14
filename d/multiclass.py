# Practical 2A - Multiclass Classification using Deep Neural Network
# Dataset: UCI Letter Recognition (letter-recognition.data)
# Aim: Classify 26 English letters (A-Z) from pixel features

# ============================================================
# Cell 1 - Import Libraries
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# Cell 2 - Define Column Names
# ============================================================
columns = ["lettr", "x-box", "y-box", "width", "height", "onpix",
           "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
           "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

# ============================================================
# Cell 3 - Load Dataset
# ============================================================
df = pd.read_csv('letter-recognition.data', names=columns)

# ============================================================
# Cell 4 - Preview Dataset
# ============================================================
print(df.head())

# ============================================================
# Cell 5 - Prepare Features and Target
# ============================================================
x = df.drop("lettr", axis=1).values
y = df["lettr"].values

# ============================================================
# Cell 6 - Check Shapes and Unique Classes
# ============================================================
print("x shape:", x.shape)
print("y shape:", y.shape)
print("Unique classes:", np.unique(y))

# ============================================================
# Cell 7 - Train-Test Split
# ============================================================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# ============================================================
# Cell 8 - Define Class Names
# ============================================================
class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
               'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# ============================================================
# Cell 9 - Normalize Features (values are 0-15, so divide by 16)
# ============================================================
x_train = x_train / 16
x_test  = x_test  / 16

# ============================================================
# Cell 10 - Encode Labels (A-Z -> 0-25)
# ============================================================
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test  = encoder.transform(y_test)

# ============================================================
# Cell 11 - Import Keras
# ============================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ============================================================
# Cell 12 - Build Model
# ============================================================
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(16,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(26, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ============================================================
# Cell 13 - Train Model
# ============================================================
model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))

# ============================================================
# Cell 14 - Predict on All Test Data
# ============================================================
predictions = model.predict(x_test)

# ============================================================
# Cell 15 - Single Sample Prediction
# ============================================================
index = 10
Pred = model.predict(x_test[index].reshape(1, -1))
final_value = np.argmax(Pred)
print("Actual label  :", y_test[index])
print("Predicted label:", final_value)
print("Class (A-Z)   :", class_names[final_value])
