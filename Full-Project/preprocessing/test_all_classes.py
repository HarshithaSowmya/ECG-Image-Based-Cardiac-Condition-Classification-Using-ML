from ecg_signal_extractor import process_ecg


test_images = {
    "MI": "dataset/MI/MI(1).jpg",
    "AHB": "dataset/AHB/HB(1).jpg",
    "PMI": "dataset/PMI/PMI(1).jpg",
    "NORMAL": "dataset/NORMAL/Normal(1).jpg",
}


for label, image_path in test_images.items():

    print("\n-----------------------------")
    print("Class:", label)
    print("Image:", image_path)

    try:
        features = process_ecg(image_path)

        if features is None:
            print("❌ Signal extraction failed")
        else:
            print("✅ Feature shape:", features.shape)
            print("Min:", features.min())
            print("Max:", features.max())

    except Exception as e:
        print("❌ Error:", e)