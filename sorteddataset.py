import os
import shutil

# Original dataset (keep it unchanged)
dataset_path = r"C:\Users\poovi\AI ResQ\dataset_balanced"

# New organized dataset
organized_path = r"C:\Users\poovi\AI ResQ\dataset_organized"

# Create main folders
distress_path = os.path.join(organized_path, "distress")
non_distress_path = os.path.join(organized_path, "non_distress")

os.makedirs(distress_path, exist_ok=True)
os.makedirs(non_distress_path, exist_ok=True)

# Counters for renaming
distress_count = 1
non_distress_count = 1

# Define which folders are distress vs non-distress (based on original folder names)
distress_folders = ["distress"]        # original distress folder name
non_distress_folders = ["non_distress"]  # original non-distress folder name

print("Organizing audio files...\n")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith((".wav", ".mp3", ".ogg")):
            old_path = os.path.join(root, file)
            
            folder_name = os.path.basename(root).lower()
            
            if folder_name in distress_folders:
                new_name = f"distress_{distress_count:03d}.wav"
                new_path = os.path.join(distress_path, new_name)
                distress_count += 1
            elif folder_name in non_distress_folders:
                new_name = f"non_distress_{non_distress_count:03d}.wav"
                new_path = os.path.join(non_distress_path, new_name)
                non_distress_count += 1
            else:
                # Skip any unknown folders
                continue

            shutil.copy(old_path, new_path)  # copy to new folder, keep original safe
            print(f"{file} ➜ {new_name}")

print("\n✅ All audio files copied and renamed successfully!")