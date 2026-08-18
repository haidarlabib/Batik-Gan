"""
Module: dataset.py
Deskripsi: Dataset loader PyTorch untuk citra motif batik tidak berlabel:
           - Membaca file citra dari disk secara efisien
           - Menerapkan pembagian Grouped Train/Test Split (80:20) berdasarkan Base ID
           - Menjamin TIDAK TERJADI data leakage antara train dan test set
"""

import os
import sys
import re
import random
from typing import List, Tuple, Optional
from PIL import Image

# Tambahkan root path proyek agar import src.* selalu berhasil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
try:
    from src.preprocessing import get_transforms
except (ImportError, ModuleNotFoundError):
    from preprocessing import get_transforms

class BatikDataset(Dataset):
    """PyTorch Dataset untuk citra motif batik tidak berlabel dengan in-memory caching."""
    
    def __init__(self, file_paths: List[str], transform=None, preload: bool = True):
        self.file_paths = file_paths
        self.transform = transform if transform is not None else get_transforms()
        self.preload = preload
        self.cached_tensors = []
        
        if self.preload:
            for path in self.file_paths:
                with Image.open(path) as img:
                    img_rgb = img.convert('RGB')
                if self.transform:
                    t = self.transform(img_rgb)
                else:
                    t = transforms.ToTensor()(img_rgb)
                self.cached_tensors.append(t)
        
    def __len__(self) -> int:
        return len(self.file_paths)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.preload:
            return self.cached_tensors[idx]
            
        path = self.file_paths[idx]
        with Image.open(path) as img:
            img_rgb = img.convert('RGB')
        
        if self.transform:
            tensor = self.transform(img_rgb)
        else:
            tensor = transforms.ToTensor()(img_rgb)
            
        return tensor

def get_train_test_file_paths(
    dataset_dir: str = "dataset",
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Membagi seluruh file dataset menjadi train dan test paths dengan Base ID grouping."""
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Direktori dataset '{dataset_dir}' tidak ditemukan.")
        
    all_files = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Kelompokkan file berdasarkan Base ID (cth: '12' untuk '12a.png' dan '12b.png')
    groups = {}
    ungrouped = []
    
    for f in all_files:
        full_path = os.path.join(dataset_dir, f)
        m = re.match(r"^(\d+)([a-zA-Z]+)\.\w+$", f)
        if m:
            base_id = int(m.group(1))
            if base_id not in groups:
                groups[base_id] = []
            groups[base_id].append(full_path)
        else:
            ungrouped.append(full_path)
            
    # Lakukan shuffling pada level grup base_id
    rng = random.Random(seed)
    base_ids = sorted(list(groups.keys()))
    rng.shuffle(base_ids)
    
    split_point = int(len(base_ids) * train_ratio)
    train_base_ids = base_ids[:split_point]
    test_base_ids = base_ids[split_point:]
    
    train_files = []
    for bid in train_base_ids:
        train_files.extend(groups[bid])
        
    test_files = []
    for bid in test_base_ids:
        test_files.extend(groups[bid])
        
    # Bagi ungrouped files jika ada
    if ungrouped:
        rng.shuffle(ungrouped)
        u_split = int(len(ungrouped) * train_ratio)
        train_files.extend(ungrouped[:u_split])
        test_files.extend(ungrouped[u_split:])
        
    return sorted(train_files), sorted(test_files)

def get_train_test_loaders(
    dataset_dir: str = "dataset",
    batch_size: int = 64,
    image_size: int = 64,
    train_ratio: float = 0.8,
    preload: bool = True,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, BatikDataset, BatikDataset]:
    """Mengembalikan DataLoader untuk Training dan Test/Reference set."""
    train_paths, test_paths = get_train_test_file_paths(dataset_dir, train_ratio=train_ratio, seed=seed)
    
    tf = get_transforms(image_size=image_size)
    train_dataset = BatikDataset(train_paths, transform=tf, preload=preload)
    test_dataset = BatikDataset(test_paths, transform=tf, preload=preload)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset

if __name__ == "__main__":
    t_loader, v_loader, t_data, v_data = get_train_test_loaders(batch_size=32)
    print(f"Total Dataset Images  : {len(t_data) + len(v_data)}")
    print(f"Training Set Size     : {len(t_data)} gambar ({len(t_loader)} batches)")
    print(f"Test/Ref Set Size     : {len(v_data)} gambar ({len(v_loader)} batches)")
    sample_batch = next(iter(t_loader))
    print(f"Sample Batch Shape    : {sample_batch.shape}")
    print(f"Tensor Min / Max Value: {sample_batch.min().item():.2f} / {sample_batch.max().item():.2f}")
