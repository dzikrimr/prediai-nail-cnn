import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/kuku_model.h5")

# Load gambar
img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediksi
prediction = model.predict(img_array)[0][0]

label = "prediabet" if prediction >= 0.5 else "non_diabet"
confidence = prediction if prediction >= 0.5 else 1 - prediction

print(f"Prediksi: {label} ({confidence*100:.2f}% yakin)")
