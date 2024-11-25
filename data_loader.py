import os
import random
import yaml
import torch
from PIL import Image
from torch.utils.data import Dataset

# Charger le fichier params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

class SportsDatasetSubset(Dataset):
    def __init__(self, data_dir, feature_extractor, split="train", selected_classes=None):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.split = split
        self.img_paths = []
        self.labels = []

        all_classes = sorted(os.listdir(data_dir))
        excluded_classes = params['prepare'].get("excluded_classes", [])
        included_classes = [cls for cls in all_classes if cls not in excluded_classes]

        if selected_classes:
            classes = [cls for cls in included_classes if cls in selected_classes]
        else:
            classes = included_classes

        for idx, cls_name in enumerate(classes):
            cls_folder = os.path.join(data_dir, cls_name)
            img_files = [f for f in os.listdir(cls_folder) if f.endswith(".jpg")]

            # Limiter le nombre d'images par classe pour équilibrer
            max_images_per_class = params['prepare'].get("max_images_per_class")
            if max_images_per_class:
                img_files = img_files[:max_images_per_class]

            # Appliquer le fractionnement basé sur sample_fraction
            sample_fraction = params['prepare'].get("sample_fraction", 1.0)
            sample_size = int(len(img_files) * sample_fraction)
            img_files = random.sample(img_files, sample_size)

            # Calcul de l'échantillonnage en fonction du split
            train_split = 1 - params['prepare'].get("split", 0.2)
            num_train = int(len(img_files) * train_split)
            if self.split == "train":
                sampled_files = img_files[:num_train]
            else:
                sampled_files = img_files[num_train:]

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