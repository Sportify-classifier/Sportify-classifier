import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from data_loader import SportsDatasetSubset
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import sys  # Import ajouté pour accéder aux arguments de ligne de commande
import random

# Charger les paramètres depuis params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

def train_model(data_dir, model_output_dir, selected_classes):
    model_name = params['model']['name']
    num_classes = len(selected_classes)
    num_folds = params['train']['num_folds']
    all_losses_train = []
    all_losses_val = []

    feature_extractor = EfficientNetImageProcessor.from_pretrained(model_name)

    # Charger les données
    dataset = SportsDatasetSubset(data_dir=data_dir, feature_extractor=feature_extractor, split="train", selected_classes=selected_classes)
    dataset_size = len(dataset)
    fold_size = dataset_size // num_folds
    indices = np.arange(dataset_size)

    for fold in range(num_folds):
        if num_folds > 1:
            print(f"Cross-validation pour le fold {fold + 1}/{num_folds}")
        else:
            print("Sans cross-validation")

        if num_folds == 1:
            train_indices = np.arange(dataset_size)
            val_indices = []
        else:
            val_indices = indices[fold * fold_size:(fold + 1) * fold_size]
            train_indices = np.setdiff1d(indices, val_indices)

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        if len(val_indices) > 0:
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size=params['train']['batch_size'], shuffle=False)
        else:
            val_dataloader = None

        train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=params['train']['batch_size'], shuffle=True)

        model = EfficientNetForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
        optimizer = Adam(model.parameters(), lr=params['train']['lr'])
        model.train()

        losses_train = []
        losses_val = []

        for epoch in range(params['train']['epochs']):
            # Entraînement
            model.train()
            epoch_loss_train = 0
            for batch in tqdm(train_dataloader):
                inputs = batch['pixel_values'].squeeze(1)
                labels = batch['labels']

                outputs = model(pixel_values=inputs, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss_train += loss.item()
            losses_train.append(epoch_loss_train / len(train_dataloader))

            # Validation
            if val_dataloader:
                model.eval()
                epoch_loss_val = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        inputs = batch['pixel_values'].squeeze(1)
                        labels = batch['labels']

                        outputs = model(pixel_values=inputs, labels=labels)
                        loss = outputs.loss
                        epoch_loss_val += loss.item()
                losses_val.append(epoch_loss_val / len(val_dataloader))

            if losses_val:
                print(f"Fold {fold + 1}, Époque {epoch + 1}, Perte entraînement : {losses_train[-1]:.4f}, Perte validation : {losses_val[-1]:.4f}")
            else:
                print(f"Époque {epoch + 1}, Perte entraînement : {losses_train[-1]:.4f}")

        all_losses_train.append(losses_train)
        all_losses_val.append(losses_val)

    # Moyennes des pertes sur tous les folds
    mean_losses_train = np.mean(all_losses_train, axis=0)
    mean_losses_val = np.mean(all_losses_val, axis=0) if all_losses_val else np.array([])

    # Graphique des pertes
    epochs = range(1, params['train']['epochs'] + 1)
    plt.plot(epochs, mean_losses_train, label="Perte entraînement")
    if mean_losses_val.size > 0:
        plt.plot(epochs, mean_losses_val, label="Perte validation")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.title(
        "Perte moyenne par époque (avec validation)" if mean_losses_val.size > 0 else "Perte moyenne par époque (sans validation)")
    plt.legend()

    # Enregistrer le graphique dans le dossier du modèle
    os.makedirs(model_output_dir, exist_ok=True)  # Création du dossier du modèle si nécessaire
    plt.savefig(os.path.join(model_output_dir, 'loss_curve.png'))
    plt.close()

    # Sauvegarder le modèle
    model.save_pretrained(model_output_dir)
    feature_extractor.save_pretrained(model_output_dir)
    print(f"Modèle sauvegardé dans {model_output_dir}")

    return model

# Ajout d'une fonction principale pour exécuter l'entraînement
if __name__ == "__main__":
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) != 3:
        print("Usage: python train.py <data_prepared_dir> <model_output_dir>")
        sys.exit(1)

    data_prepared_dir = sys.argv[1]
    model_output_dir = sys.argv[2]

    # Fixer les graines pour la reproductibilité
    random_seed = params['train'].get("seed", 42)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Spécifier le chemin des données
    data_dir = os.path.join(data_prepared_dir, 'train')  # Chemin des données d'entraînement

    # Sélectionner les classes à inclure
    selected_classes = sorted(os.listdir(data_dir))

    # Appeler la fonction d'entraînement
    train_model(data_dir=data_dir, model_output_dir=model_output_dir, selected_classes=selected_classes)