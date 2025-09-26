import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model (kuku atau lidah)
model = load_model("models/kuku_model.h5")

# Fungsi preprocessing gambar
def prepare_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Fungsi mapping risiko dan ciri
def map_risk_factors(label, prediction):
    if label == "non_diabet":
        # Normal → Tidak teridentifikasi
        return "Tidak teridentifikasi", [
            "Kuku berwarna merata",
            "Tekstur halus",
            "Permukaan rata"
        ]
    else:
        confidence = prediction
        if confidence <= 0.7:
            # Risiko sedang
            return "Risiko sedang", [
                "Tekstur tidak rata",
                "Perubahan warna ringan",
                "Permukaan sedikit bergelombang"
            ]
        else:
            # Risiko tinggi
            return "Risiko tinggi", [
                "Perubahan warna kuku",
                "Permukaan kuku tidak normal",
                "Tanda Onychomycosis / Paronychia"
            ]

# Ambil path gambar dari argumen
if len(sys.argv) < 2:
    print("Gunakan: python predict_risk.py path/to/image.jpg")
    sys.exit(1)

img_path = sys.argv[1]
img_array = prepare_image(img_path)

# Prediksi
prediction = model.predict(img_array)[0][0]
label = "prediabet" if prediction >= 0.5 else "non_diabet"

risk_level, risk_factors = map_risk_factors(label, prediction)
confidence = prediction if label == "prediabet" else 1 - prediction

# Tampilkan hasil
print(f"Prediksi: {label} ({confidence*100:.2f}% yakin)")
print(f"Tingkat Risiko: {risk_level}")
print("Ciri yang terdeteksi:")
for f in risk_factors:
    print(f"- {f}")
