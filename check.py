import os

train_furnished = "data/train/furnished"
train_unfurnished = "data/train/unfurnished"

val_furnished = "data/val/furnished"
val_unfurnished = "data/val/unfurnished"

print("Train Furnished Exists:", os.path.exists(train_furnished))
print("Train Unfurnished Exists:", os.path.exists(train_unfurnished))
print("Val Furnished Exists:", os.path.exists(val_furnished))
print("Val Unfurnished Exists:", os.path.exists(val_unfurnished))
