# evaluate.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
test_dir = "data/test"

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

model_path = "models/kuku_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model {model_path} tidak ditemukan. Jalankan train.py dulu!")

model = tf.keras.models.load_model(model_path)

pred_probs = model.predict(test_data)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

print("\n=== 📊 Classification Report (Test Set) ===")
report = classification_report(
    test_data.classes,
    pred_labels,
    target_names=list(test_data.class_indices.keys()),
    digits=4
)
print(report)

conf_mat = confusion_matrix(test_data.classes, pred_labels)
print("\n=== 🔢 Confusion Matrix ===")
print(conf_mat)

classes = list(test_data.class_indices.keys())

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (Test Set - Kuku Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("evaluation_confusion_matrix_kuku.png")
plt.close()

report_dict = classification_report(
    test_data.classes,
    pred_labels,
    target_names=classes,
    output_dict=True
)

metrics = ["precision", "recall", "f1-score"]
values = [
    [report_dict[cls][metric] for cls in classes]
    for metric in metrics
]

plt.figure(figsize=(7, 4))
x = np.arange(len(classes))
width = 0.25

for i, metric in enumerate(metrics):
    plt.bar(x + i * width, values[i], width, label=metric.capitalize())

plt.xticks(x + width, classes)
plt.ylim(0, 1)
plt.title("Precision, Recall, and F1-score per Class (Kuku Model)")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation_metrics_bar_kuku.png")
plt.close()

overall_acc = report_dict["accuracy"] * 100
print(f"\n✅ Overall Accuracy: {overall_acc:.2f}%")
print("\n📊 Visualisasi evaluasi disimpan sebagai:")
print(" - evaluation_confusion_matrix_kuku.png")
print(" - evaluation_metrics_bar_kuku.png")
