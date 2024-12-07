import sys
import torch
import yaml
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from torch.utils.data import DataLoader, Subset
from utils import SportsDataset, create_versioned_dir

# Charger les paramètres depuis params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

def train_model(data_dir, model_output_dir):
    model_name = params['model']['name']
    num_folds = params['train']['num_folds']
    all_losses_train = []
    all_losses_val = []

    # Initialiser le feature extractor
    feature_extractor = EfficientNetImageProcessor.from_pretrained(model_name)

    # Charger le dataset
    dataset = SportsDataset(data_dir=data_dir, feature_extractor=feature_extractor)
    dataset_size = len(dataset)
    fold_size = dataset_size // num_folds
    indices = np.arange(dataset_size)

    for fold in range(num_folds):
        print(f"Entraînement pour le fold {fold + 1}/{num_folds}")

        # Diviser les indices pour le train et la validation
        val_indices = indices[fold * fold_size:(fold + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        # Créer les DataLoaders
        train_dataloader = DataLoader(train_subset, batch_size=params['train']['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=params['train']['batch_size'], shuffle=False)

        # Initialiser le modèle
        model = EfficientNetForImageClassification.from_pretrained(
            model_name,
            num_labels=len(dataset.classes),
            ignore_mismatched_sizes=True
        )
        optimizer = Adam(model.parameters(), lr=params['train']['lr'])
        model.train()

        losses_train = []
        losses_val = []

        for epoch in range(params['train']['epochs']):
            # Boucle d'entraînement
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

            # Boucle de validation
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

            print(f"Fold {fold + 1}, Époque {epoch + 1}, Perte entraînement : {losses_train[-1]:.4f}, Perte validation : {losses_val[-1]:.4f}")

        all_losses_train.append(losses_train)
        all_losses_val.append(losses_val)

    # Calculer la perte moyenne sur tous les folds
    mean_losses_train = np.mean(all_losses_train, axis=0)
    mean_losses_val = np.mean(all_losses_val, axis=0)

    # Plot les pertes
    epochs = range(1, params['train']['epochs'] + 1)
    plt.plot(epochs, mean_losses_train, label="Perte entraînement")
    plt.plot(epochs, mean_losses_val, label="Perte validation")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.title("Perte moyenne par époque")
    plt.legend()

    # Enregistrer le graphique
    os.makedirs(model_output_dir, exist_ok=True)
    plt.savefig(os.path.join(model_output_dir, 'loss_curve.png'))
    plt.close()

    # Sauvegarder le modèle
    model.save_pretrained(model_output_dir)
    feature_extractor.save_pretrained(model_output_dir)
    print(f"Modèle sauvegardé dans {model_output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <data_prepared_dir> <model_output_base_dir>")
        sys.exit(1)

    data_prepared_dir = sys.argv[1]
    model_output_base_dir = sys.argv[2]

    # Fixer les graines pour la reproductibilité
    random_seed = params['train'].get("seed", 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    data_dir = os.path.join(data_prepared_dir, 'train')

    # Créer un dossier versionné pour le modèle
    model_output_dir = create_versioned_dir(model_output_base_dir, "model")

    # Appeler la fonction d'entraînement
    train_model(data_dir=data_dir, model_output_dir=model_output_dir)