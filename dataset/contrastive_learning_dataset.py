from torchvision.transforms import transforms
from src.dataset.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from src.dataset.view_generator import ContrastiveLearningViewGenerator, WatermarkViewGenerator
from src.dataset.sicap import SicapDataset
import os

from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from PIL import Image
import numpy as np

class ContrastiveLearningDataset:
    """
    A class to handle datasets for contrastive learning tasks, providing data augmentation 
    pipelines and dataset loading functionality.
    Methods
    -------
    __init__(root_folder):
        Initializes the dataset object with the root folder path.
    get_simclr_pipeline_transform(size, s=1):
        Static method that returns a set of data augmentation transformations as described 
        in the SimCLR paper. Includes random cropping, flipping, color jittering, grayscale 
        conversion, Gaussian blur, and tensor conversion.
    get_imagenet_transform(size, s=1):
        Static method that returns a simple data augmentation pipeline for ImageNet, 
        including random cropping and tensor conversion.
    get_dataset(name, n_views):
        Retrieves the training dataset for the specified dataset name. Supports CIFAR-10, 
        STL-10, SVHN, and ImageNet. Applies the SimCLR data augmentation pipeline and 
        supports multiple views for contrastive learning.
    get_test_dataset(name, n_views):
        Retrieves the test dataset for the specified dataset name. Supports CIFAR-10, 
        STL-10, SVHN, and ImageNet. Applies the SimCLR data augmentation pipeline and 
        supports multiple views for contrastive learning.
    Attributes
    ----------
    root_folder : str
        The root directory where datasets are stored or will be downloaded.
    """

    def __init__(self, root_folder):
        print("_"*100)
        print("Using contrastive dataset")
        print("_"*100)
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.ToTensor()])
        return data_transforms


    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"{self.root_folder}/stl10", split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(f"{self.root_folder}/SVHN",
                                                          split='train',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(
                                                                  32),
                                                              n_views),
                                                          download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet256/",
                              split='train',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_simclr_pipeline_transform(
                                      224),
                                  n_views))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()
    def get_test_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32), 
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"{self.root_folder}/stl10", split='test',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(f"{self.root_folder}/SVHN",
                                                        split='test',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(
                                                                32),
                                                            n_views),
                                                        download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet256/",
                              split='val',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_simclr_pipeline_transform(
                                      224),
                                  n_views))
                          }


        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()
        
        
class RegularDataset:
    """
    A class for handling datasets used in contrastive learning tasks. This class provides methods to 
    retrieve training and testing datasets with appropriate transformations applied.
    Attributes:
    ----------
    root_folder : str
        The root directory where datasets are stored.
    Methods:
    -------
    get_simclr_pipeline_transform(size, s=1):
        Static method to create a basic transformation pipeline for SimCLR-style contrastive learning.
        Parameters:
            size (int): The size of the image.
            s (float): Scale factor for transformations (default is 1).
        Returns:
            torchvision.transforms.Compose: A composed transformation pipeline.
    get_imagenet_transform(size, s=1):
        Static method to create a transformation pipeline specifically for ImageNet dataset.
        Parameters:
            size (int): The size of the image.
            s (float): Scale factor for transformations (default is 1).
        Returns:
            torchvision.transforms.Compose: A composed transformation pipeline.
    get_dataset(name, n_views):
        Retrieves the training dataset based on the provided name and applies the appropriate transformations.
        Parameters:
            name (str): The name of the dataset ('cifar10', 'stl10', 'svhn', 'imagenet').
            n_views (int): Number of views to generate for contrastive learning.
        Returns:
            torchvision.datasets: The specified training dataset with transformations applied.
        Raises:
            Exception: If the dataset name is not recognized.
    get_test_dataset(name, n_views):
        Retrieves the testing dataset based on the provided name and applies the appropriate transformations.
        Parameters:
            name (str): The name of the dataset ('cifar10', 'stl10', 'svhn', 'imagenet').
            n_views (int): Number of views to generate for contrastive learning.
        Returns:
            torchvision.datasets: The specified testing dataset with transformations applied.
        Raises:
            Exception: If the dataset name is not recognized.
    """

    def __init__(self, root_folder):
        print("_"*100)
        print("Using regular dataset")
        print("_"*100)
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        data_transforms = transforms.Compose([transforms.Resize(size),
                                              transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True, 
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32), n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"{self.root_folder}/stl10", split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(f"{self.root_folder}/SVHN",
                                                        split='train',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(
                                                                32),
                                                            n_views),
                                                        download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root=f"{self.root_folder}/imagenet_pytorch/",
                              split='train',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_imagenet_transform(
                                      224),
                                  n_views))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()
    def get_test_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                              transform=ContrastiveLearningViewGenerator(  
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"{self.root_folder}/stl10", split='test',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(f"{self.root_folder}/SVHN",
                                                        split='test',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(
                                                                32),
                                                            n_views),
                                                        download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet_pytorch/",
                              split='val',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_imagenet_transform(
                                      224),
                                  n_views))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()




class CustomCifarDataset(datasets.CIFAR10):
    """
    Custom CIFAR10 dataset class that allows for different transformations for victim and surrogate views.
    """

    def __init__(self,
        root: Union[str, Path],
        train: bool = True,
        victim_transform: Optional[Callable] = None,
        surrogate_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(CustomCifarDataset, self).__init__(root, train=train, transform=victim_transform,
                                                 target_transform=target_transform, download=download)
        self.victim_transform = victim_transform
        self.surrogate_transform = surrogate_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.victim_transform is not None:
            victim_img = self.victim_transform(img)

        if self.surrogate_transform is not None:
            target_img = self.surrogate_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return victim_img, target_img, target
    
class CustomSTL10Dataset(datasets.STL10):
    """
    Custom STL10 dataset class that allows for different transformations for victim and surrogate views.
    """

    def __init__(self,
        root: Union[str, Path],
        split: str = 'unlabeled',
        victim_transform: Optional[Callable] = None,
        surrogate_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(CustomSTL10Dataset, self).__init__(root, split=split, transform=victim_transform,
                                                 target_transform=target_transform, download=download)
        self.victim_transform = victim_transform
        self.surrogate_transform = surrogate_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.victim_transform is not None:
            victim_img = self.victim_transform(img)

        if self.surrogate_transform is not None:
            target_img = self.surrogate_transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return victim_img, target_img, target
    
class CustomSVHNDataset(datasets.SVHN):
    """
    Custom STL10 dataset class that allows for different transformations for victim and surrogate views.
    """

    def __init__(self,
        root: Union[str, Path],
        split: str = 'train',
        victim_transform: Optional[Callable] = None,
        surrogate_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(CustomSVHNDataset, self).__init__(root, split=split, transform=victim_transform,
                                                 target_transform=target_transform, download=download)
        self.victim_transform = victim_transform
        self.surrogate_transform = surrogate_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.victim_transform is not None:
            victim_img = self.victim_transform(img)

        if self.surrogate_transform is not None:
            target_img = self.surrogate_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return victim_img, target_img, target


class CustomRegularDataset(RegularDataset):
    """
    Implementation of Regular Dataset that returns 2 views, 1st for Victim and 2nd for Surrogate.
    """

    def __init__(self, root_folder, victim_transform, surrogate_transform):
        super(CustomRegularDataset, self).__init__(root_folder)
        print("_"*100)
        print("Using Custom Regular dataset")
        print("_"*100)
        self.root_folder = root_folder

        self.victim_transform = victim_transform
        self.surrogate_transform = surrogate_transform

    def get_dataset(self, name, n_views, **kwargs):

        percent_data = kwargs.get('percent_data', 1.0)
        corrupt_pct = kwargs.get('corrupt_pct', 0.0)
        clip_version = kwargs.get('clip_version', "CLIP")

        valid_datasets = {'cifar10': lambda: CustomCifarDataset(self.root_folder, train=True, 
                                                              victim_transform=self.victim_transform,
                                                              surrogate_transform=self.surrogate_transform,
                                                              download=True),

                          'stl10': lambda: CustomSTL10Dataset(f"{self.root_folder}/stl10", split='unlabeled',
                                                          victim_transform=self.victim_transform,
                                                          surrogate_transform=self.surrogate_transform,
                                                          download=True),

                          'svhn': lambda: CustomSVHNDataset(f"{self.root_folder}/SVHN",
                                                        split='train',
                                                        victim_transform=self.victim_transform,
                                                        surrogate_transform=self.surrogate_transform,
                                                        download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root=f"{self.root_folder}/imagenet_pytorch/",
                              split='train',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_imagenet_transform(
                                      224),
                                  n_views)),
                            'sicap': lambda: SicapDataset(
                                root=self.root_folder,
                                train=True,
                                victim_transform=self.victim_transform,
                                surrogate_transform=self.surrogate_transform,
                                percent_data=percent_data, 
                                corrupt_pct=corrupt_pct,
                                clip_version=clip_version)
                          }
        
        if name == 'imagenet':
            raise NotImplementedError("CustomRegularDataset does not support ImageNet dataset.")

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()
        
    def get_test_dataset(self, name, n_views, **kwargs):

        percent_data = kwargs.get('percent_data', 1.0)
        corrupt_pct = kwargs.get('corrupt_pct', 0.0)
        clip_version = kwargs.get('clip_version', "CLIP")
        
        valid_datasets = {'cifar10': lambda: CustomCifarDataset(self.root_folder, train=False, 
                                                              victim_transform=self.victim_transform,
                                                              surrogate_transform=self.surrogate_transform,
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"{self.root_folder}/stl10", split='test',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(f"{self.root_folder}/SVHN",
                                                        split='test',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(
                                                                32),
                                                            n_views),
                                                        download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet_pytorch/",
                              split='val',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_imagenet_transform(
                                      224),
                                  n_views)),
                           'sicap': lambda: SicapDataset(
                                root=self.root_folder,
                                train=False,
                                victim_transform=self.victim_transform,
                                surrogate_transform=self.surrogate_transform,
                                percent_data=percent_data, 
                                corrupt_pct=corrupt_pct,
                                clip_version=clip_version)
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()

class WatermarkDataset:
    """
    WatermarkDataset is a class designed to handle datasets with specific transformations 
    and augmentations for contrastive learning tasks. It provides methods to apply 
    custom transformations and retrieve datasets with multiple views.
    Attributes:
        root_folder (str): The root directory where datasets are stored.
    Methods:
        __init__(root_folder):
            Initializes the WatermarkDataset instance with the specified root folder.
        get_transform():
            Static method that returns a list of two data transformations. 
            Each transformation includes random rotation and conversion to tensor.
        get_imagenet_transform(size, s=1):
            Static method that returns a list of two data transformations 
            specifically for ImageNet. Each transformation includes random resized 
            cropping, random rotation, and conversion to tensor.
        get_dataset(name, n_views):
            Retrieves the specified dataset with the applied transformations and 
            the desired number of views. Supports 'cifar10', 'stl10', 'svhn', and 'imagenet'.
            Raises an exception if the dataset name is invalid.
    """
    
    def __init__(self, root_folder):
        print("_"*100)
        print("Using watermark dataset")
        print("_"*100)
        self.root_folder = root_folder

    @staticmethod
    def get_transform():
        data_transform1 = transforms.Compose([transforms.RandomRotation(degrees=(0, 180)),
                                              transforms.ToTensor()])
        data_transform2 = transforms.Compose([transforms.RandomRotation(degrees=(180, 360)),
                                              transforms.ToTensor()])
        return [data_transform1, data_transform2]

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ToTensor()])
        data_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomRotation(degrees=(180, 360)),
             transforms.ToTensor()])
        return [data_transform1, data_transform2]

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=WatermarkViewGenerator(
                                                                  self.get_transform(),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"{self.root_folder}/stl10", split='unlabeled',
                                                          transform=WatermarkViewGenerator(
                                                              self.get_transform(),
                                                              n_views),
                                                          download=True),
                          'svhn': lambda: datasets.SVHN(
                              f"{self.root_folder}/SVHN",
                              split='test',
                              transform=WatermarkViewGenerator(
                                  self.get_transform(),
                                  n_views),
                              download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet_pytorch/",
                              split='val',
                              transform=WatermarkViewGenerator(
                                  self.get_imagenet_transform(
                                      32),
                                  n_views))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()