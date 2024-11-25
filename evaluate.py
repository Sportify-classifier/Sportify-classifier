import torch
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from data_loader import SportsDatasetSubset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import sys
import json
import shutil

# Charger les paramètres depuis params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

def evaluate_model(model_dir, data_dir, output_dir, selected_classes):
    feature_extractor = EfficientNetImageProcessor.from_pretrained(model_dir)
    model = EfficientNetForImageClassification.from_pretrained(model_dir)
    model.eval()

    test_dataset = SportsDatasetSubset(
        data_dir=data_dir,
        feature_extractor=feature_extractor,
        split="test",
        selected_classes=selected_classes
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params['train']['batch_size'],
        shuffle=False
    )

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

    # Calcul des métriques
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')  # F1-score pondéré
    classif_report = classification_report(all_labels, all_predictions, target_names=selected_classes, output_dict=True)

    # Sauvegarder les métriques dans un fichier JSON
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': classif_report
    }
    os.makedirs(output_dir, exist_ok=True)  # Création du dossier de sortie si nécessaire
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métriques sauvegardées dans {os.path.join(output_dir, 'metrics.json')}")

    # Afficher les métriques
    print(f'Précision globale : {accuracy * 100:.2f}%')
    print(f'F1-score (pondéré) : {f1 * 100:.2f}%')

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=selected_classes, yticklabels=selected_classes, cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.title('Matrice de confusion')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))  # Enregistrer la matrice de confusion
    plt.close()
    print(f"Matrice de confusion sauvegardée dans {os.path.join(output_dir, 'confusion_matrix.png')}")

    # Copier la courbe de perte depuis le dossier du modèle vers le dossier d'évaluation
    loss_curve_src = os.path.join(model_dir, 'loss_curve.png')
    loss_curve_dst = os.path.join(output_dir, 'loss_curve.png')
    if os.path.exists(loss_curve_src):
        shutil.copyfile(loss_curve_src, loss_curve_dst)
        print(f"Courbe de perte copiée dans {loss_curve_dst}")
    else:
        print("Courbe de perte non trouvée dans le dossier du modèle.")

# Ajout d'une fonction principale pour exécuter l'évaluation
if __name__ == "__main__":
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) != 4:
        print("Usage: python evaluate.py <model_dir> <data_prepared_dir> <evaluation_output_dir>")
        sys.exit(1)

    model_dir = sys.argv[1]
    data_prepared_dir = sys.argv[2]
    output_dir = sys.argv[3]

    # Spécifier le chemin des données
    data_dir = os.path.join(data_prepared_dir, 'test')  # Chemin des données de test

    # Sélectionner les classes à inclure
    selected_classes = sorted(os.listdir(data_dir))

    # Appeler la fonction d'évaluation
    evaluate_model(model_dir=model_dir, data_dir=data_dir, output_dir=output_dir, selected_classes=selected_classes)