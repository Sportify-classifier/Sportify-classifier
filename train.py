import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from data_loader import SportsDatasetSubset
from config import config
import numpy as np
import matplotlib.pyplot as plt

def train_model(data_dir, selected_classes):
    model_name = config['model_name']
    num_classes = len(selected_classes)
    num_folds = config['num_folds']
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
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
        if len(val_indices) > 0:
            val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
        else:
            val_dataloader = None

        model = EfficientNetForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
        optimizer = Adam(model.parameters(), lr=config['learning_rate'])
        model.train()

        losses_train = []
        losses_val = []

        for epoch in range(config['num_epochs']):
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
    epochs = range(1, config['num_epochs'] + 1)
    plt.plot(epochs, mean_losses_train, label="Perte entraînement")
    if mean_losses_val.size > 0:
        plt.plot(epochs, mean_losses_val, label="Perte validation")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.title("Perte moyenne par époque (avec validation)" if mean_losses_val.size > 0 else "Perte moyenne par époque (sans validation)")
    plt.legend()
    plt.show()

    return model


