import os
import numpy as np
import pandas as pd

from preprocessing.ecg_signal_extractor import process_ecg


DATASET_DIR = "dataset"
OUTPUT_FILE = "training/ecg_features.csv"

CLASS_LABELS = {
    "NORMAL": 0,
    "AHB": 1,
    "MI": 2,
    "PMI": 3
}


rows = []

total = sum(
    len([
        f for f in os.listdir(os.path.join(DATASET_DIR, class_name))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    for class_name in CLASS_LABELS
)

processed = 0
failed = 0

print(f"Total images: {total}")


for class_name, label in CLASS_LABELS.items():

    class_dir = os.path.join(DATASET_DIR, class_name)

    files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\nProcessing {class_name}: {len(files)} images")

    for filename in files:

        image_path = os.path.join(class_dir, filename)

        try:
            features = process_ecg(image_path)

            if features is None or len(features) != 3060:
                print(f"❌ Failed: {image_path}")
                failed += 1
                continue

            row = features.tolist()
            row.append(label)
            row.append(class_name)
            row.append(filename)

            rows.append(row)

            processed += 1

            if processed % 25 == 0:
                print(f"Processed: {processed}/{total}")

        except Exception as e:
            print(f"❌ Error: {image_path}")
            print("   ", e)
            failed += 1


# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

feature_columns = [f"feature_{i}" for i in range(3060)]

columns = feature_columns + [
    "label",
    "class_name",
    "filename"
]

df = pd.DataFrame(rows, columns=columns)

os.makedirs("training", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n================================")
print("Dataset creation complete")
print("================================")

print("Successful:", processed)
print("Failed:", failed)
print("Shape:", df.shape)

print("\nClass distribution:")
print(df["class_name"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)