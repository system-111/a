# Practical 3B - Convolutional Neural Network (CNN) for Fashion Classification
# Dataset: Fashion MNIST (fashion-mnist_train.csv / fashion-mnist_test.csv)
# Aim: Classify 10 categories of fashion clothing using CNN

# ============================================================
# Cell 1 - Import Libraries
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Cell 2 - Define Class Names
# ============================================================
class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boot']

# ============================================================
# Cell 3 - Load Dataset from CSV
# ============================================================
df1 = pd.read_csv("fashion-mnist_train.csv")
df2 = pd.read_csv("fashion-mnist_test.csv")

# ============================================================
# Cell 4 - Prepare Training Data
# ============================================================
x_train = df1.drop("label", axis=1).values
y_train = df1["label"].values
print("x_train shape:", x_train.shape)

# ============================================================
# Cell 5 - Prepare Test Data
# ============================================================
x_test = df2.drop("label", axis=1).values
y_test = df2["label"].values

# ============================================================
# Cell 6 - Reshape to 28x28 Images (for visualization)
# ============================================================
x_train = x_train.reshape(60000, 28, 28)
x_test  = x_test.reshape(10000, 28, 28)

# ============================================================
# Cell 7 - Visualize a Sample Image
# ============================================================
plt.imshow(x_train[0], cmap='gray')
plt.show()

# ============================================================
# Cell 8 - Normalize Pixel Values to 0-1
# ============================================================
x_train = x_train / 255
x_test  = x_test  / 255

# ============================================================
# Cell 9 - Reshape to 4D for CNN (add channel dimension)
# ============================================================
x_train = x_train.reshape(60000, 28, 28, 1)
x_test  = x_test.reshape(10000, 28, 28, 1)

# ============================================================
# Cell 10 - Import Keras Layers
# ============================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# ============================================================
# Cell 11 - Build CNN Model
# ============================================================
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ============================================================
# Cell 12 - Train Model
# ============================================================
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# ============================================================
# Cell 13 - Predict on All Test Data
# ============================================================
predictions = model.predict(x_test)

# ============================================================
# Cell 14 - Single Sample Prediction
# ============================================================
index = 10
print("Prediction probabilities:", predictions[index])
final_value = np.argmax(predictions[index])
print("Actual:      ", y_test[index])
print("Predicted:   ", final_value)
print("Class Label: ", class_name[final_value])

# ============================================================
# Cell 15 - Visualize the Test Sample
# ============================================================
plt.imshow(x_test[index])
plt.show()

# ============================================================
# Cell 16 - Evaluate Model
# ============================================================
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:     ", loss)
print("Accuracy: ", accuracy * 100)
