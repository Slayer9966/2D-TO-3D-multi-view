import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils import binvox_rw
import torchvision.transforms as transforms  # ADD THIS LINE
import random  # ADD THIS LINE


class ShapeNetDataset(Dataset):
    def __init__(self, render_root, voxel_root, transform=None, num_views=13, image_size=256, is_training=True):  # Add is_training parameter
        self.transform = transform
        self.object_ids = []
        self.data_pairs = []
        self.num_views = num_views
        self.image_size = image_size
        self.is_training = is_training  # Add this line
        
        # Add augmentation transforms - INSERT THIS BLOCK
        if is_training and transform is None:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=25, fill=255),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
                transforms.RandomPerspective(distortion_scale=0.08, p=0.3),
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif not is_training and transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Traverse all category folders (keep your existing code)
        for category in os.listdir(render_root):
            category_render_path = os.path.join(render_root, category)
            category_voxel_path = os.path.join(voxel_root, category)
            
            if not os.path.isdir(category_render_path):
                continue
                
            # Go inside each object_id folder
            for obj_id in os.listdir(category_render_path):
                view_path = os.path.join(category_render_path, obj_id)
                voxel_path = os.path.join(category_voxel_path, obj_id, "model.binvox")
                
                if not os.path.isdir(view_path):
                    continue
                    
                if os.path.exists(voxel_path):
                    self.data_pairs.append((view_path, voxel_path))
                    self.object_ids.append(obj_id)  
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        view_path, voxel_path = self.data_pairs[idx]
        obj_id = self.object_ids[idx]
        view_images = sorted(glob.glob(os.path.join(view_path, "*.png")))
        
        # ADD THIS LINE - Randomly shuffle views during training
        if self.is_training:
            random.shuffle(view_images)
        
        # Take only first 13 views (or fewer if less are available)
        selected_views = view_images[:self.num_views]
        
        views = []
        for img_path in selected_views:
            img = Image.open(img_path).convert("RGB").resize((self.image_size, self.image_size))
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
            views.append(img)
        
        # REPLACE this section for smarter padding:
        while len(views) < self.num_views:
            if self.is_training and len(views) > 0:
                # Create augmented version of random existing view
                random_view_idx = random.randint(0, len(views) - 1)
                views.append(views[random_view_idx].clone())
            else:
                # For test set, just repeat the last view
                views.append(views[-1].clone())
        
        views = torch.stack(views)  # Now always [13, 3, 256, 256]
        
        # ADD THIS BLOCK - Random view dropout during training
        if self.is_training:
            if random.random() < 0.1:  # 10% chance
                num_to_drop = random.randint(1, min(3, self.num_views // 4))
                drop_indices = random.sample(range(self.num_views), num_to_drop)
                for drop_idx in drop_indices:
                    views[drop_idx] = torch.zeros_like(views[drop_idx])
        
        # Load voxel (keep your existing code)
        with open(voxel_path, "rb") as f:
            voxel_model = binvox_rw.read(f)
        voxel = torch.tensor(voxel_model.data, dtype=torch.float32).unsqueeze(0)
        
        return views, voxel, obj_id
