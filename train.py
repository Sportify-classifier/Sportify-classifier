import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from data_loader import SportsDatasetSubset
from config import config

def train_model(data_dir, selected_classes):
    # Charger le modèle et l'extracteur de caractéristiques
    model_name = config['model_name']
    num_classes = len(selected_classes)
    model = EfficientNetForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
    feature_extractor = EfficientNetImageProcessor.from_pretrained(model_name)

    # Charger les données
    train_dataset = SportsDatasetSubset(
        data_dir=data_dir, 
        feature_extractor=feature_extractor, 
        sample_fraction=config['sample_fraction'],
        selected_classes=selected_classes
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialiser l'optimiseur
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    model.train()

    # Boucle d'entraînement
    for epoch in range(config['num_epochs']):
        for batch in tqdm(train_dataloader):
            inputs = batch['pixel_values'].squeeze(1)
            labels = batch['labels']

            # Forward pass
            outputs = model(pixel_values=inputs, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Époque {epoch+1}, perte : {loss.item()}")
    return model
