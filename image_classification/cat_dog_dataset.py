"""Custom pytorch Dataset for binary image classification."""
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class CatDogDataset(Dataset):
    """Create Pytorch Dataset for cat/dog binary classification."""

    def __init__(self, file_list, file_dir, transform):
        """
        Init method.

        Parameters
        ----------
        file_list : List[str]
            List with names of the files. Must include dog or cat in the name.
        file_dir : str
            Directory path.
        transform : Callable
            Transformations to apply to tensors.
        """
        self.file_list = file_list
        self.file_dir = file_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.file_dir, self.file_list[idx]))
        if "dog" in self.file_list[idx]:
            label = 1.0
        else:
            label = 0.0

        img = self.transform(img)
        return {"image": img, "target": torch.tensor([label])}
