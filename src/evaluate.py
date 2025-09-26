import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path dataset test
test_dir = "data/test"
img_size = (224, 224)
batch_size = 32

# Data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",   # ✅ binary classification
    shuffle=False
)

# Load model
model = tf.keras.models.load_model("models/kuku_model.h5")

# Evaluasi akurasi & loss
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
print(f"✅ Test Loss: {loss:.4f}")
