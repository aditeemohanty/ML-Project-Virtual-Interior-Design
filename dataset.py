from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class FurnishedUnfurnishedDataset(Dataset):
    def __init__(self, root_unfurnished, root_furnished, transform=None):
        self.root_unfurnished = root_unfurnished
        self.root_furnished = root_furnished
        self.transform = transform

        if not os.path.exists(self.root_unfurnished) or not os.path.exists(self.root_furnished):
            raise FileNotFoundError("One or both dataset folders do not exist. Check the paths.")

        self.unfurnished_images = os.listdir(root_unfurnished)
        self.furnished_images = os.listdir(root_furnished)

        if len(self.unfurnished_images) == 0 or len(self.furnished_images) == 0:
            raise FileNotFoundError("One or both dataset folders are empty. Add images to proceed.")

        self.length_dataset = max(len(self.unfurnished_images), len(self.furnished_images))
        self.unfurnished_len = len(self.unfurnished_images)
        self.furnished_len = len(self.furnished_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        unfurnished_img = self.unfurnished_images[index % self.unfurnished_len]
        furnished_img = self.furnished_images[index % self.furnished_len]

        unfurnished_path = os.path.join(self.root_unfurnished, unfurnished_img)
        furnished_path = os.path.join(self.root_furnished, furnished_img)

        unfurnished_img = np.array(Image.open(unfurnished_path).convert("RGB"))
        furnished_img = np.array(Image.open(furnished_path).convert("RGB"))

        # Print the shapes of images before transformations
        print(f"Unfurnished image shape: {unfurnished_img.shape}")
        print(f"Furnished image shape: {furnished_img.shape}")

        if self.transform:
            augmentations = self.transform(image=unfurnished_img, image0=furnished_img)
            unfurnished_img = augmentations["image"]
            furnished_img = augmentations["image0"]

        return unfurnished_img, furnished_img
