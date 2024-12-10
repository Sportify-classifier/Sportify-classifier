import os
import sys
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from torch.utils.data import DataLoader
from utils import SportsDataset, log_class_metrics_to_wandb, log_artifact_to_wandb, update_best_accuracy, update_wandb_tags
import wandb
import json

# Charger les paramètres depuis params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

def get_latest_model_dir(model_versions_dir):
    subdirs = [
        os.path.join(model_versions_dir, d) for d in os.listdir(model_versions_dir)
        if os.path.isdir(os.path.join(model_versions_dir, d))
    ]
    if not subdirs:
        raise Exception(f"Aucun dossier de modèle trouvé dans {model_versions_dir}")
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir

def create_versioned_dir(base_dir, prefix):
    # Créer un dossier avec un horodatage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(versioned_dir, exist_ok=True)
    return versioned_dir

def plot_precision_recall_curves(all_labels, all_predictions, classes, output_dir):
    # Binariser les labels et les prédictions
    y_test = label_binarize(all_labels, classes=range(len(classes)))
    y_score = label_binarize(all_predictions, classes=range(len(classes)))

    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision = average_precision_score(y_test[:, i], y_score[:, i])
        plt.step(recall, precision, where='post', label=f'{classes[i]} (AP={average_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbe Precision-Recall')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'))
    plt.close()
    print(f"Courbes Precision-Recall sauvegardées dans {os.path.join(output_dir, 'precision_recall_curves.png')}")

def generate_html_report(metrics, output_dir):
    # Préparer les métriques au format HTML
    metrics_html = '''
    <h2>Métriques</h2>
    <pre>{}</pre>
    '''.format(json.dumps(metrics, indent=4))

    # Liste des graphiques à inclure
    plots = [
        'loss_curve.png',
        'confusion_matrix.png',
        'normalized_confusion_matrix.png',
        'precision_recall_curves.png'
    ]

    # Construire le contenu HTML
    html_content = f"""
    <html>
    <head>
        <title>Rapport d'Évaluation du Modèle</title>
    </head>
    <body>
        <h1>Rapport d'Évaluation du Modèle</h1>
        {metrics_html}
    """

    for plot in plots:
        if os.path.exists(os.path.join(output_dir, plot)):
            html_content += f"""
            <h2>{plot.replace('_', ' ').title()}</h2>
            <img src="{plot}" alt="{plot}" style="max-width:100%; height:auto;">
            """

    html_content += """
    </body>
    </html>
    """

    # Sauvegarder le rapport HTML
    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(html_content)
    print(f"Rapport HTML sauvegardé dans {os.path.join(output_dir, 'report.html')}")

def evaluate_model(model_dir, data_dir, evaluation_versions_dir):
    # Créer un dossier versionné pour les évaluations
    output_dir = create_versioned_dir(evaluation_versions_dir, 'model')

    # Initialiser le feature extractor et le modèle
    feature_extractor = EfficientNetImageProcessor.from_pretrained(model_dir)
    model = EfficientNetForImageClassification.from_pretrained(model_dir)
    model.eval()

    # Charger le dataset de test
    test_dataset = SportsDataset(data_dir=data_dir, feature_extractor=feature_extractor)
    test_dataloader = DataLoader(test_dataset, batch_size=params['train']['batch_size'], shuffle=False)

    all_labels = []
    all_predictions = []

    # Boucle d'évaluation
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch['pixel_values'].squeeze(1)
            labels = batch['labels']
            outputs = model(pixel_values=inputs)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Initialiser W&B pour l'évaluation
    # Lire l'identifiant du run depuis le fichier
    with open("wandb_run_id.txt", "r") as f:
        run_id = f.read().strip()


    # Calcul des métriques
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    cohen_kappa = cohen_kappa_score(all_labels, all_predictions)
    mcc = matthews_corrcoef(all_labels, all_predictions)

    # Vérifier si le modèle actuel est le meilleur
    is_best_model = update_best_accuracy(accuracy)

    # Rejoindre le run existant
    wandb.init(
        project="sports-classification",
        id=run_id,
        resume="allow"
    )

    if wandb.run.id:
        print(f"Run ID W&B actif : {wandb.run.id}")
    else:
        print("Erreur : aucun ID de run actif.")

    # Mise à jour des tags dans W&B
    update_wandb_tags(
        project_name="sports-classification",
        current_run_id=wandb.run.id,
        is_best_model=is_best_model
    )

    # Rapport de classification
    classif_report = classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        output_dict=True,
        zero_division=0
    )

    # Sauvegarder les métriques
    os.makedirs(output_dir, exist_ok=True)
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_score_weighted': f1,
        'cohen_kappa': cohen_kappa,
        'mcc': mcc,
        'classification_report': classif_report
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métriques sauvegardées dans {os.path.join(output_dir, 'metrics.json')}")

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.title('Matrice de confusion')
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Matrice de confusion sauvegardée dans {confusion_matrix_path}")

    # Matrice de confusion normalisée
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.title('Matrice de confusion normalisée')
    plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix.png'))
    plt.close()
    print(f"Matrice de confusion normalisée sauvegardée dans {os.path.join(output_dir, 'normalized_confusion_matrix.png')}")

    # Courbes Precision-Recall
    plot_precision_recall_curves(all_labels, all_predictions, test_dataset.classes, output_dir)

    # Ajouter les tags W&B
    wandb_tags = ["last_evaluation"]
    if is_best_model:
        wandb_tags.append("best_model")
    wandb.run.tags += tuple(wandb_tags)

    # Loguer l'image de la matrice de confusion dans W&B
    wandb.log({"Confusion Matrix Heatmap": wandb.Image(confusion_matrix_path)})

    # Logguez les métriques et les courbes
    wandb.log({"accuracy": accuracy, "f1-score": f1, "precision": precision, "recall": recall})

    # Extraire les noms des classes
    classes = test_dataset.classes

    # Logger les métriques de précision, rappel et F1-score par classe
    log_class_metrics_to_wandb(classif_report, classes) 

    # Copier la courbe de perte depuis le dossier du modèle
    loss_curve_src = os.path.join(model_dir, 'loss_curve.png')
    loss_curve_dst = os.path.join(output_dir, 'loss_curve.png')
    if os.path.exists(loss_curve_src):
        from shutil import copyfile
        copyfile(loss_curve_src, loss_curve_dst)
        print(f"Courbe de perte copiée dans {loss_curve_dst}")
    else:
        print("Courbe de perte non trouvée dans le dossier du modèle.")

    # Générer le rapport HTML
    generate_html_report(metrics, output_dir)

    # Log des artefacts d'évaluation dans W&B
    log_artifact_to_wandb("evaluation_results", "evaluation", output_dir)

    # Terminer le run W&B
    wandb.finish()

    # Copier les outputs dans un répertoire fixe pour DVC
    fixed_output_dir = 'evaluation_outputs'
    os.makedirs(fixed_output_dir, exist_ok=True)
    # Liste des fichiers à copier
    output_files = [
        'metrics.json',
        'loss_curve.png',
        'confusion_matrix.png',
        'normalized_confusion_matrix.png',
        'precision_recall_curves.png',
        'report.html'
    ]
    from shutil import copyfile
    for file_name in output_files:
        src_file = os.path.join(output_dir, file_name)
        dst_file = os.path.join(fixed_output_dir, file_name)
        if os.path.exists(src_file):
            if os.path.abspath(src_file) != os.path.abspath(dst_file):
                copyfile(src_file, dst_file)
                print(f"Fichier {file_name} copié dans {fixed_output_dir}")
            else:
                print(f"Le fichier source et destination sont identiques pour {file_name}, pas de copie effectuée.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python evaluate.py <model_versions_dir> <data_prepared_dir> <evaluation_versions_dir>")
        sys.exit(1)

    model_versions_dir = sys.argv[1]
    data_prepared_dir = sys.argv[2]
    evaluation_versions_dir = sys.argv[3]

    # Obtenir le dernier dossier de modèle
    model_dir = get_latest_model_dir(model_versions_dir)
    data_dir = os.path.join(data_prepared_dir, 'test')

    # Appeler la fonction d'évaluation
    evaluate_model(model_dir=model_dir, data_dir=data_dir, evaluation_versions_dir=evaluation_versions_dir)
