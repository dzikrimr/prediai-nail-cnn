import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path dataset validasi
valid_dir = "data/valid"
img_size = (224, 224)
batch_size = 32

# Data generator
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_gen = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# Load model
model = tf.keras.models.load_model("models/kuku_model.h5")

# Prediksi
y_pred_prob = model.predict(valid_gen)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()
y_true = valid_gen.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=valid_gen.class_indices.keys(),
            yticklabels=valid_gen.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Kuku Model")
plt.savefig("models/confusion_matrix.png")
plt.show()

print("📂 Confusion matrix saved: models/confusion_matrix.png")
