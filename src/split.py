import os
import shutil
import random

base_dir = "data"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

val_ratio = 0.2

os.makedirs(valid_dir, exist_ok=True)

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    valid_class_path = os.path.join(valid_dir, class_name)
    os.makedirs(valid_class_path, exist_ok=True)

    images = os.listdir(class_path)
    random.shuffle(images)

    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]

    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(valid_class_path, img)
        shutil.move(src, dst)

    print(f"✅ {len(val_images)} files moved from {class_path} → {valid_class_path}")
