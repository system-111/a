# Practical 3A - Convolutional Neural Network (CNN) for Plant Disease Detection
# Dataset: Tomato Disease Dataset (10 classes, images in ./archive/tomato/)
# Aim: Detect tomato plant diseases from leaf images using CNN

# ============================================================
# Cell 1 - Import Libraries
# ============================================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ============================================================
# Cell 2 - Set Dataset Paths
# ============================================================
train_dir = "./archive/tomato/train"
val_dir   = "./archive/tomato/val"

# ============================================================
# Cell 3 - Set Image Size and Batch Size
# ============================================================
img_size   = 224
batch_size = 32

# ============================================================
# Cell 4 - Load Training Data with Normalization
# ============================================================
train_datagen  = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# ============================================================
# Cell 5 - Load Validation Data
# ============================================================
val_datagen  = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# ============================================================
# Cell 6 - View Class Names
# ============================================================
print(list(train_generator.class_indices))

# ============================================================
# Cell 7 - Import Keras Layers
# ============================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# ============================================================
# Cell 8 - Build CNN Model
# ============================================================
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(train_generator.num_classes, activation='softmax'))

model.summary()

# ============================================================
# Cell 9 - Compile Model
# ============================================================
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ============================================================
# Cell 10 - Train Model
# ============================================================
model.fit(train_generator, epochs=2, validation_data=val_generator)

# ============================================================
# Cell 11 - Evaluate Model
# ============================================================
loss, accuracy = model.evaluate(val_generator)
print("Loss:", loss)
print("Accuracy:", accuracy * 100)

# ============================================================
# Cell 12 - Load and Preprocess a Single Test Image
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

img_path  = "./archive/tomato/val/Tomato___Septoria_leaf_spot/0a25f893-1b5f-4845-baa1-f68ac03d96ac___Matt.S_CG 7863.jpg"
img       = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255

# ============================================================
# Cell 13 - Predict Class of the Test Image
# ============================================================
prediction = model.predict(img_array)
class_name = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ============================================================
# Cell 14 - Display Prediction Result
# ============================================================
predicted_class = np.argmax(prediction)
print("Prediction probabilities:", prediction)
print("Predicted class index:", predicted_class)
print("Class Name:", class_name[predicted_class])
