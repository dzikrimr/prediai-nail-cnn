import os

base_dir = "data"  # sesuaikan dengan lokasi dataset kamu
splits = ["train", "valid", "test"]
classes = os.listdir(os.path.join(base_dir, "train"))

# hitung jumlah file per kelas dan split
data_count = {cls: {split: 0 for split in splits} for cls in classes}

for split in splits:
    for cls in classes:
        path = os.path.join(base_dir, split, cls)
        if os.path.exists(path):
            data_count[cls][split] = len(os.listdir(path))

# tampilkan hasil
total_all = 0
train_total = valid_total = test_total = 0

for cls in classes:
    train = data_count[cls]["train"]
    valid = data_count[cls]["valid"]
    test = data_count[cls]["test"]
    total = train + valid + test
    total_all += total
    train_total += train
    valid_total += valid
    test_total += test
    print(f"✅ {cls}: {train} train, {valid} valid, {test} test (total {total})")

print(f"\n🔢 Total semua data: {total_all}\n")

# hitung proporsi
train_pct = (train_total / total_all) * 100
valid_pct = (valid_total / total_all) * 100
test_pct = (test_total / total_all) * 100

print(f"📈 Proporsi aktual:")
print(f"Train: {train_total} ({train_pct:.1f}%)")
print(f"Valid: {valid_total} ({valid_pct:.1f}%)")
print(f"Test : {test_total} ({test_pct:.1f}%)")
