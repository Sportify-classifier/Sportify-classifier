import torch
from transformers import EfficientNetImageProcessor
from data_loader import SportsDatasetSubset
from config import config
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, data_dir, selected_classes):

    feature_extractor = EfficientNetImageProcessor.from_pretrained(config['model_name'])
    model.eval()
    # Charger les données de test
    test_dataset = SportsDatasetSubset(
        data_dir=data_dir, 
        feature_extractor=feature_extractor, 
        sample_fraction=config['test_sample_fraction'],
        selected_classes=selected_classes
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch['pixel_values'].squeeze(1)
            labels = batch['labels']
            outputs = model(pixel_values=inputs)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculer la précision globale
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    print(f'Précision globale : {accuracy * 100:.2f}%')

    # Générer le rapport de classification avec des métriques moyennes
    report = classification_report(all_labels, all_predictions, target_names=selected_classes, output_dict=True)
    print("Rapport de classification :")
    print(classification_report(all_labels, all_predictions, target_names=selected_classes))

    # Générer et visualiser la matrice de confusion
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=selected_classes, yticklabels=selected_classes, cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.title('Matrice de confusion')
    plt.show()
