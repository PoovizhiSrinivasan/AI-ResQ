import os

dataset_path = r"C:\Users\poovi\AI ResQ\dataset_balanced"

print("Scanning dataset...\n")

for root, dirs, files in os.walk(dataset_path):

    # folder name will be the label
    label = os.path.basename(root)

    count = 1

    for file in files:

        if file.lower().endswith((".wav", ".mp3", ".ogg")):

            old_path = os.path.join(root, file)

            new_name = f"{label}_{count:02d}.wav"

            new_path = os.path.join(root, new_name)

            try:
                os.rename(old_path, new_path)
                print(f"{file} ➜ {new_name}")
                count += 1
            except Exception as e:
                print("Error:", e)

print("\n✅ All audio files renamed successfully!")