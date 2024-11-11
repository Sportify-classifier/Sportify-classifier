import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from config import config

# Classe de chargement des données
class SportsDatasetSubset(Dataset):
    def __init__(self, data_dir, feature_extractor, sample_fraction=config['sample_fraction'], selected_classes=None):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        if selected_classes:
            classes = selected_classes
        else:
            classes = sorted(os.listdir(data_dir))
        self.img_paths = []
        self.labels = []
        
        # Parcourir chaque classe et sélectionner un sous-ensemble d'images
        for idx, cls_name in enumerate(classes):
            cls_folder = os.path.join(data_dir, cls_name)
            img_files = [f for f in os.listdir(cls_folder) if f.endswith(".jpg")]
            num_samples = max(1, int(len(img_files) * sample_fraction))
            sampled_files = random.sample(img_files, num_samples)
            
            for img_file in sampled_files:
                self.img_paths.append(os.path.join(cls_folder, img_file))
                self.labels.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs