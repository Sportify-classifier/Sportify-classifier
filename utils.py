import os
import random
import yaml
import shutil
from PIL import Image
from torch.utils.data import Dataset
import torch

# Charger le fichier params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

def prepare_data(input_data_dir, output_data_dir):
    # Fixer la seed pour la reproductibilité
    random_seed = params['prepare'].get("seed", 42)
    random.seed(random_seed)

    # Obtenir toutes les classes disponibles
    all_classes = sorted(os.listdir(input_data_dir))
    excluded_classes = params['prepare'].get("excluded_classes", [])
    # Inclure seulement les classes non exclues
    included_classes = [cls for cls in all_classes if cls not in excluded_classes]

    # Limiter le nombre de classes si spécifié
    num_classes = params['train'].get("num_classes")
    if num_classes and num_classes < len(included_classes):
        selected_classes = random.sample(included_classes, num_classes)
    else:
        selected_classes = included_classes

    # Préparer les dossiers de sortie pour train et test
    train_output_dir = os.path.join(output_data_dir, 'train')
    test_output_dir = os.path.join(output_data_dir, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    for cls_name in selected_classes:
        cls_folder = os.path.join(input_data_dir, cls_name)
        img_files = [f for f in os.listdir(cls_folder) if f.endswith(".jpg")]

        # Limiter le nombre d'images par classe pour équilibrer
        max_images_per_class = params['prepare'].get("max_images_per_class")
        if max_images_per_class:
            img_files = img_files[:max_images_per_class]

        # Mélanger les images avant de les séparer
        random.shuffle(img_files)
        train_split = 1 - params['prepare'].get("split", 0.2)
        num_train = int(len(img_files) * train_split)
        train_files = img_files[:num_train]
        test_files = img_files[num_train:]

        # Copier les fichiers dans les dossiers correspondants
        for split, files in [('train', train_files), ('test', test_files)]:
            split_class_dir = os.path.join(output_data_dir, split, cls_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img_file in files:
                src_path = os.path.join(cls_folder, img_file)
                dst_path = os.path.join(split_class_dir, img_file)
                shutil.copyfile(src_path, dst_path)

    print(f"Données préparées et enregistrées dans {output_data_dir}")

class SportsDataset(Dataset):
    def __init__(self, data_dir, feature_extractor):
        # Initialiser le dataset
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.img_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))

        # Parcourir chaque classe et collecter les images
        for idx, cls_name in enumerate(self.classes):
            cls_folder = os.path.join(data_dir, cls_name)
            img_files = [f for f in os.listdir(cls_folder) if f.endswith(".jpg")]
            for img_file in img_files:
                self.img_paths.append(os.path.join(cls_folder, img_file))
                self.labels.append(idx)

    def __len__(self):
        # Retourne la taille du dataset
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Obtenir l'image et son label
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs

def create_versioned_dir(base_dir, prefix):
    # Créer un dossier versionné avec un horodatage
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(versioned_dir, exist_ok=True)
    return versioned_dir