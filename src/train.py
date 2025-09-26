import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Path dataset
train_dir = "data/train"

# Image size & batch
img_size = (224, 224)
batch_size = 32

# Augmentasi data + validasi split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Train generator
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",   # ✅ binary classification
    subset="training"
)

# Validation generator
valid_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",   # ✅ binary classification
    subset="validation"
)

# Build model CNN 
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # freeze feature extractor

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')   # ✅ sigmoid for binary
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_gen, validation_data=valid_gen, epochs=15)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/kuku_model.h5")
print("✅ Model trained & saved.")
