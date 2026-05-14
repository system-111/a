# Practical 2B - Binary Classification using Deep Neural Network
# Dataset: IMDB Movie Reviews (loaded from Keras)
# Aim: Classify movie reviews as Positive or Negative

# ============================================================
# Cell 1 - Import IMDB Dataset
# ============================================================
from tensorflow.keras.datasets import imdb

# ============================================================
# Cell 2 - Load Data (top 10000 most frequent words only)
# ============================================================
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# ============================================================
# Cell 3 - Check Shapes
# ============================================================
print("Train shape: ", x_train.shape)
print("Test shape: ",  x_test.shape)
print("Train labels shape: ", y_train.shape)
print("Test labels shape: ",  y_test.shape)

# ============================================================
# Cell 4 - View a Raw Review (encoded as word indices)
# ============================================================
print(x_train[1])
print("Label (0=Negative, 1=Positive):", y_train[1])

# ============================================================
# Cell 5 - Get Word Index and Build Reverse Lookup
# ============================================================
vocab = imdb.get_word_index()
print("Index of 'the':", vocab['the'])

Class_name = ['Negative', 'Positive']
reverse_index = dict([(value, key) for (key, value) in vocab.items()])

# ============================================================
# Cell 6 - Decode a Review Back to Text
# ============================================================
def decode(review):
    text = ""
    for i in review:
        text = text + reverse_index[i]
        text += " "
    return text

print(decode(x_train[1]))

# ============================================================
# Cell 7 - Check Review Lengths (before padding)
# ============================================================
def showlen():
    print("Length of x_train[0]:", len(x_train[0]))
    print("Length of x_train[1]:", len(x_train[1]))
    print("Length of x_test[0]: ", len(x_test[0]))
    print("Length of x_test[1]: ", len(x_test[1]))

showlen()

# ============================================================
# Cell 8 - Pad Sequences to Fixed Length of 256
# (padding='post' adds zeros at end; value=0 is standard padding token)
# ============================================================
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, value=0, padding='post', maxlen=256)
x_test  = pad_sequences(x_test,  value=0, padding='post', maxlen=256)

# ============================================================
# Cell 9 - Check Lengths After Padding
# ============================================================
showlen()

# ============================================================
# Cell 10 - Import Keras Layers
# ============================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

# ============================================================
# Cell 11 - Build Model
# ============================================================
model = Sequential()
model.add(Embedding(10000, 16))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ============================================================
# Cell 12 - Train Model
# ============================================================
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# ============================================================
# Cell 13 - Single Sample Prediction
# ============================================================
import numpy as np
predicted_value = model.predict(np.expand_dims(x_test[10], 0))
print("Raw prediction:", predicted_value)

if predicted_value > 0.5:
    final_value = 1
else:
    final_value = 0

print("Predicted class:", final_value)
print("Sentiment:", Class_name[final_value])
