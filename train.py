import subprocess
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
from utils import SportsDataset, create_versioned_dir, calculate_accuracy, log_metrics_to_wandb
import wandb

# Charger les paramètres depuis params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

def train_model(data_dir, model_output_dir):
    model_name = params['model']['name']
    num_folds = params['train']['num_folds']
    all_losses_train = []
    all_losses_val = []

    # Lire l'identifiant du run depuis le fichier ou le créer s'il n'existe pas
    run_id_file = "wandb_run_id.txt"
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
    else:
        run_id = wandb.util.generate_id()
        with open(run_id_file, "w") as f:
            f.write(run_id)

    # Rejoindre le run existant
    wandb.init(
        project="sports-classification",
        id=run_id,
        resume="allow"
    )

    # Initialiser le feature extractor
    feature_extractor = EfficientNetImageProcessor.from_pretrained(model_name)

    # Charger le dataset
    dataset = SportsDataset(data_dir=data_dir, feature_extractor=feature_extractor)
    dataset_size = len(dataset)
    fold_size = dataset_size // num_folds
    indices = np.arange(dataset_size)

    for fold in range(1, num_folds + 1):
        print(f"Entraînement pour le fold {fold}/{num_folds}")

        # Diviser les indices pour le train et la validation
        val_indices = indices[(fold - 1) * fold_size:fold * fold_size]
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

        # Initialiser les listes pour les pertes par époque dans un fold
        losses_train = []
        losses_val = []

        for epoch in range(params['train']['epochs']):
            # Boucle d'entraînement
            model.train()
            epoch_loss_train = 0
            epoch_accuracy_train = 0
            for batch in tqdm(train_dataloader):
                inputs = batch['pixel_values'].squeeze(1)
                labels = batch['labels']

                outputs = model(pixel_values=inputs, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss_train += loss.item()
                epoch_accuracy_train += calculate_accuracy(outputs.logits, labels)

            # Moyenne des métriques pour cette epoch
            train_loss = epoch_loss_train / len(train_dataloader)
            train_accuracy = epoch_accuracy_train / len(train_dataloader)

            # Boucle de validation
            model.eval()
            epoch_loss_val = 0
            epoch_accuracy_val = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = batch['pixel_values'].squeeze(1)
                    labels = batch['labels']

                    outputs = model(pixel_values=inputs, labels=labels)
                    loss = outputs.loss
                    epoch_loss_val += loss.item()
                    epoch_accuracy_val += calculate_accuracy(outputs.logits, labels)

            # Moyenne des métriques pour cette epoch
            val_loss = epoch_loss_val / len(val_dataloader)
            val_accuracy = epoch_accuracy_val / len(val_dataloader)

            # Ajouter les pertes pour cette epoch aux listes locales
            losses_train.append(train_loss)
            losses_val.append(val_loss)

            # Logger les métriques dans W&B
            log_metrics_to_wandb(epoch, train_loss, val_loss, train_accuracy, val_accuracy)

        # Ajouter les pertes pour ce fold aux listes globales
        all_losses_train.append(losses_train)
        all_losses_val.append(losses_val)

    # Calculer la moyenne des pertes d'entraînement et de validation sur tous les folds
    mean_losses_train = np.mean(all_losses_train, axis=0)  # Moyenne des pertes d'entraînement
    mean_losses_val = np.mean(all_losses_val, axis=0)      # Moyenne des pertes de validation

    # Plot les pertes
    epochs = range(1, params['train']['epochs'] + 1)
    plt.plot(epochs, mean_losses_train, label="Perte entraînement")
    plt.plot(epochs, mean_losses_val, label="Perte validation")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.title("Perte moyenne par époque")
    plt.legend()

    # Enregistrer le graphique
    loss_plot_path = os.path.join(model_output_dir, 'loss_curve.png')

    os.makedirs(model_output_dir, exist_ok=True)
    plt.savefig(loss_plot_path)
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