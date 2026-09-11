import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
class Pix2PixDataset(Dataset):
    """
    Dataset class for loading paired images (e.g., side-by-side combined images).
    Assumes each image file in the directory has width = 2 * height, 
    where the left half is image A and the right half is image B.
    """
    def __init__(self, root_dir, direction='a2b', is_train=True, target_size=256):
        super(Pix2PixDataset, self).__init__()
        self.root_dir = root_dir
        self.direction = direction
        self.is_train = is_train
        self.target_size = target_size
        
        # Get list of images
        self.image_files = sorted([
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if len(self.image_files) == 0:
            print(f"Warning: No images found in {root_dir}")
        # Static transform for test/validation (just resizing and normalizing)
        self.transform_test = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def __len__(self):
        return len(self.image_files)
    def _transform(self, img_a, img_b):
        # 1. Resize to target_size + jitter margin if training
        if self.is_train:
            jitter_size = int(self.target_size * 1.12)  # e.g., 286 for 256
            img_a = TF.resize(img_a, (jitter_size, jitter_size), Image.BICUBIC)
            img_b = TF.resize(img_b, (jitter_size, jitter_size), Image.BICUBIC)
            
            # 2. Random crop back to target_size
            i, j, h, w = transforms.RandomCrop.get_params(
                img_a, output_size=(self.target_size, self.target_size)
            )
            img_a = TF.crop(img_a, i, j, h, w)
            img_b = TF.crop(img_b, i, j, h, w)
            
            # 3. Random horizontal flip
            if random.random() > 0.5:
                img_a = TF.hflip(img_a)
                img_b = TF.hflip(img_b)
        else:
            # Test transform
            img_a = TF.resize(img_a, (self.target_size, self.target_size), Image.BICUBIC)
            img_b = TF.resize(img_b, (self.target_size, self.target_size), Image.BICUBIC)
        # Convert to Tensor and Normalize to [-1, 1]
        img_a = TF.to_tensor(img_a)
        img_b = TF.to_tensor(img_b)
        
        img_a = TF.normalize(img_a, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img_b = TF.normalize(img_b, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return img_a, img_b
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_files[index])
        
        # Open combined image
        combined_img = Image.open(img_path).convert('RGB')
        w, h = combined_img.size
        
        # Split side-by-side image
        # Left half = A, Right half = B
        img_a = combined_img.crop((0, 0, w // 2, h))
        img_b = combined_img.crop((w // 2, 0, w, h))
        
        # Apply synchronized transforms
        img_a_trans, img_b_trans = self._transform(img_a, img_b)
        
        # Select input and target based on direction
        if self.direction == 'a2b':
            input_img = img_a_trans
            target_img = img_b_trans
        else:  # b2a
            input_img = img_b_trans
            target_img = img_a_trans
            
        return input_img, target_img