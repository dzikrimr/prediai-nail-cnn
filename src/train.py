import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import numpy as np

train_dir = "data/train"
valid_dir = "data/valid"
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

valid_gen = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_gen, validation_data=valid_gen, epochs=15)

os.makedirs("models", exist_ok=True)
model.save("models/kuku_model.h5")
print("✅ Model trained & saved at models/kuku_model.h5")

valid_gen.reset()
pred_probs = model.predict(valid_gen)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

print("\n=== 📊 Validation Classification Report ===")
print(classification_report(valid_gen.classes, pred_labels, target_names=list(valid_gen.class_indices.keys())))
