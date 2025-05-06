import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

# Color mapping for segmentation masks (consistent with train_segmentation.py)
SEGMENTATION_COLORS = {
    0: [0, 0, 0],      # background
    1: [255, 0, 0],    # front
    2: [0, 253, 0],    # back
    3: [0, 0, 250],    # sleeves
    4: [253, 255, 0],  # hood
}

class SegmentationDataset(Dataset):
    """
    Dataset for image segmentation. Expects paired RGB images and color-coded mask images.
    Supports optional resizing to ensure consistent tensor shapes.

    Directory structure:
        root_dir/
            Images/      # RGB input images (*.png)
            Labels/      # RGB mask images (*.png) with matching basenames + '_mask'
    e.g., 'img01.png' <-> 'img01_mask.png'

    Args:
        root_dir (str): Path to the dataset folder.
        is_train (bool): If True, applies random horizontal flip before any resizing.
        resize (tuple[int,int], optional): If provided, resizes both image and mask to (H, W).
        transform (callable, optional): alias for image_transform (PIL->Tensor).
        image_transform (callable, optional): transforms for input images (PIL->Tensor).
        mask_transform (callable, optional): further transforms for mask PIL images (after resizing).
    """
    def __init__(self,
                 root_dir,
                 is_train=False,
                 resize=None,
                 transform=None,
                 image_transform=None,
                 mask_transform=None):
        self.image_dir = os.path.join(root_dir, 'Images')
        self.mask_dir = os.path.join(root_dir, 'Labels')
        self.image_names = sorted(f for f in os.listdir(self.image_dir) if f.endswith('.png'))
        self.is_train = is_train
        self.resize = tuple(resize) if resize is not None else None
        # Determine image transform, prioritizing 'transform' for backward compatibility
        if transform is not None:
            self.image_transform = transform
        elif image_transform is not None:
            self.image_transform = image_transform
        else:
            # Default: ToTensor + Normalize
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        # Mask transform: applied after resizing but before index conversion
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        # Random horizontal flip for augmentation
        if self.is_train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize both to fixed size if specified
        if self.resize:
            image = image.resize(self.resize[::-1], resample=Image.BILINEAR)
            mask = mask.resize(self.resize[::-1], resample=Image.NEAREST)

        # Apply mask PIL transforms (e.g., additional cropping)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        # Apply image transforms to tensor
        image = self.image_transform(image)

        # Convert mask to class indices
        mask_np = np.array(mask)
        h, w = mask_np.shape[:2]
        mask_idx = np.zeros((h, w), dtype=np.int64)
        for class_idx, color in SEGMENTATION_COLORS.items():
            matches = np.all(mask_np == color, axis=-1)
            mask_idx[matches] = class_idx

        # Create a tensor copy to ensure resizable storage
        mask_tensor = torch.from_numpy(mask_idx).clone()
        return image, mask_tensor

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to pad images and masks in a batch to the same size.
        Usage:
            DataLoader(dataset, batch_size, collate_fn=SegmentationDataset.collate_fn)
        """
        import torch.nn.functional as F
        images, masks = zip(*batch)
        # Find max height and width
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        # Pad images and masks to max dimensions (pad on right and bottom)
        padded_images = [F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])) for img in images]
        padded_masks = [F.pad(mask, (0, max_w - mask.shape[1], 0, max_h - mask.shape[0])) for mask in masks]
        # Stack into tensors
        return torch.stack(padded_images), torch.stack(padded_masks)
