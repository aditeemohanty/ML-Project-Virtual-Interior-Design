import os

# Define dataset directories
dirs = [
    "CycleGAN_dataset/data/train/furnished",
    "CycleGAN_dataset/data/train/unfurnished",
    "CycleGAN_dataset/data/val/furnished",
    "CycleGAN_dataset/data/val/unfurnished",
]

# Create directories if they don't exist
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

print("All required dataset folders have been created!")
