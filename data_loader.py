import os
import random
import yaml
import torch
from PIL import Image
from torch.utils.data import Dataset
import shutil  # Import ajouté pour copier les fichiers
import sys     # Import ajouté pour accéder aux arguments de ligne de commande

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

# Ajout d'une fonction principale pour préparer les données et les enregistrer
if __name__ == "__main__":
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) != 3:
        print("Usage: python data_loader.py <input_data_dir> <output_data_dir>")
        sys.exit(1)

    input_data_dir = sys.argv[1]
    output_data_dir = sys.argv[2]

    random_seed = params['prepare'].get("seed", 42)
    random.seed(random_seed)

    all_classes = sorted(os.listdir(input_data_dir))
    excluded_classes = params['prepare'].get("excluded_classes", [])
    included_classes = [cls for cls in all_classes if cls not in excluded_classes]

    # Limiter le nombre de classes si spécifié
    num_classes = params['train'].get("num_classes")
    if num_classes and num_classes < len(included_classes):
        selected_classes = random.sample(included_classes, num_classes)
    else:
        selected_classes = included_classes

    # Préparer les dossiers de sortie
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

        # Appliquer le fractionnement basé sur sample_fraction
        sample_fraction = params['prepare'].get("sample_fraction", 1.0)
        sample_size = int(len(img_files) * sample_fraction)
        img_files = random.sample(img_files, sample_size)

        # Mélanger les images pour le split
        random.shuffle(img_files)

        # Calcul de l'échantillonnage en fonction du split
        train_split = 1 - params['prepare'].get("split", 0.2)
        num_train = int(len(img_files) * train_split)
        train_files = img_files[:num_train]
        test_files = img_files[num_train:]

        # Préparer les dossiers de classe pour train et test
        train_class_dir = os.path.join(train_output_dir, cls_name)
        test_class_dir = os.path.join(test_output_dir, cls_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Copier les fichiers d'images dans les dossiers correspondants
        for img_file in train_files:
            src_path = os.path.join(cls_folder, img_file)
            dst_path = os.path.join(train_class_dir, img_file)
            shutil.copyfile(src_path, dst_path)
        for img_file in test_files:
            src_path = os.path.join(cls_folder, img_file)
            dst_path = os.path.join(test_class_dir, img_file)
            shutil.copyfile(src_path, dst_path)

    print(f"Données préparées et enregistrées dans {output_data_dir}")