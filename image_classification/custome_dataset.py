#################################### Custom Dataset #######################################
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

class CustomMnist(MNIST):
    

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.tensor(img).float(), torch.tensor(target)
