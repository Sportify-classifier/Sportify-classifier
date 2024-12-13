import os
import random
import yaml
import shutil
from PIL import Image
from torch.utils.data import Dataset
import torch
import wandb
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import label_binarize
import json, time



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

    # Récupérer la fraction d'échantillonnage
    sample_fraction = params['prepare'].get("sample_fraction")

    for cls_name in selected_classes:
        cls_folder = os.path.join(input_data_dir, cls_name)
        img_files = [f for f in os.listdir(cls_folder) if f.endswith(".jpg")]

        # Limiter le nombre d'images par classe pour équilibrer
        max_images_per_class = params['prepare'].get("max_images_per_class")
        if max_images_per_class:
            img_files = img_files[:max_images_per_class]

        # Appliquer la fraction d'échantillonnage
        if sample_fraction < 1.0:
            num_samples = int(len(img_files) * sample_fraction)
            img_files = random.sample(img_files, num_samples)

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


def log_class_summary_to_wandb(train_dir, test_dir, num_images_to_log=5):
    """
    Log un tableau W&B résumant la distribution des classes :
    - Nom de la classe.
    - Exemples d'images (5 par classe).
    - Nombre d'images dans le train.
    - Nombre d'images dans le test.
    Ajoute également deux graphiques pour la répartition des classes.
    """
    # Créer le tableau principal
    table_sum = wandb.Table(columns=["Class Name", "Example Images", "Train Count", "Test Count"])

    # Données pour les graphiques
    train_table = wandb.Table(columns=["Class Name", "Train Count"])
    test_table = wandb.Table(columns=["Class Name", "Test Count"])

    # Obtenir toutes les classes (présentes dans train ou test)
    all_classes = set(os.listdir(train_dir)) | set(os.listdir(test_dir))

    for cls_name in sorted(all_classes):
        # Dossiers pour train et test
        train_class_dir = os.path.join(train_dir, cls_name)
        test_class_dir = os.path.join(test_dir, cls_name)

        # Collecter les exemples d'images
        train_images = [os.path.join(train_class_dir, f) for f in os.listdir(train_class_dir) if f.endswith(".jpg")] if os.path.exists(train_class_dir) else []
        test_images = [os.path.join(test_class_dir, f) for f in os.listdir(test_class_dir) if f.endswith(".jpg")] if os.path.exists(test_class_dir) else []

        example_images = [wandb.Image(img) for img in train_images[:num_images_to_log]]


        # Ajouter les informations dans le tableau principal
        table_sum.add_data(
            cls_name,
            example_images,
            len(train_images),
            len(test_images)
        )

        # Ajouter les informations dans les tableaux pour les graphiques
        train_table.add_data(cls_name, len(train_images))
        test_table.add_data(cls_name, len(test_images))

    # Logger le tableau principal dans W&B
    wandb.log({"Class Summary": table_sum})
    print("Résumé des classes loggé dans W&B.")

    # Logger le graphique pour la répartition dans train
    wandb.log({"Class distribution for train": wandb.plot.bar(
        train_table, "Class Name", "Train Count", title="Class distribution for train"
    )})
    print("Graphique de distribution des classes (train) loggé dans W&B.")

    # Logger le graphique pour la répartition dans test
    wandb.log({"Class distribution for test": wandb.plot.bar(
        test_table, "Class Name", "Test Count", title="Class distribution for test"
    )})
    print("Graphique de distribution des classes (test) loggé dans W&B.")

def log_excluded_classes_to_wandb(input_data_dir, excluded_classes, num_images_to_log=5):
    """
    Log un tableau W&B résumant les classes exclues :
    - Nom de la classe exclue.
    - Exemples d'images (jusqu'à 5 par classe).
    - Nombre total d'images dans chaque classe exclue.
    """
    table_ex = wandb.Table(columns=["Excluded Class Name", "Example Images", "Total Images"])

    for cls_name in excluded_classes:
        cls_folder = os.path.join(input_data_dir, cls_name)

        if not os.path.exists(cls_folder):
            # Ignorer si la classe n'existe pas dans les données
            print(f"Classe exclue {cls_name} introuvable dans le répertoire {input_data_dir}.")
            continue

        # Collecter les images de la classe exclue
        img_files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.endswith(".jpg")]

        # Sélectionner les exemples d'images
        example_images = [wandb.Image(img) for img in img_files[:num_images_to_log]]

        # Ajouter les informations dans le tableau
        table_ex.add_data(
            cls_name,
            example_images,
            len(img_files)
        )

    # Logger le tableau dans W&B
    wandb.log({"Excluded Classes Summary": table_ex})
    print("Résumé des classes exclues loggé dans W&B.")

def log_metrics_to_wandb(epoch, train_loss, val_loss, train_accuracy, val_accuracy):
    """
    Log les métriques (loss et accuracy) dans Weights & Biases.
    """
    wandb.log({
        "Train Loss": train_loss,
        "Validation Loss": val_loss,
        "Train Accuracy": train_accuracy,
        "Validation Accuracy": val_accuracy,
        "Epoch": epoch + 1
    })
    print(f"Metrics logged for epoch {epoch + 1}.")

def calculate_accuracy(predictions, labels):
    """
    Calcule l'accuracy à partir des prédictions et des labels.
    """
    predicted_labels = predictions.argmax(dim=1)
    correct = (predicted_labels == labels).sum().item()
    total = labels.size(0)
    return correct / total

def log_artifact_to_wandb(artifact_name, artifact_type, artifact_dir):
    """
    Log un artefact (modèle, données, visualisations) dans W&B.
    """
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_dir(artifact_dir)
    wandb.log_artifact(artifact)

def log_class_metrics_to_wandb(classif_report, classes):
    """
    Log precision, recall, and F1-score as bar charts to W&B.

    Args:
        classif_report: Classification report as a dictionary.
        classes: List of class names.
    """
    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        # Extraire les valeurs métriques pour chaque classe
        metric_values = [classif_report[cls][metric] for cls in classes]

        # Créer un tableau pour W&B
        table = wandb.Table(data=[[cls, val] for cls, val in zip(classes, metric_values)],
                            columns=["Class", metric.capitalize()])

        # Logger en tant que bar chart
        wandb.log({
            f"{metric.capitalize()} per Class": wandb.plot.bar(
                table, "Class", metric.capitalize(), title=f"{metric.capitalize()} per Class"
            )
        })

def update_best_accuracy(current_accuracy, metrics_file="best_metrics.json"):
    """
    Met à jour la meilleure précision dans un fichier JSON.
    """
    try:
        # Convertir en float pour éviter des erreurs de type
        current_accuracy = float(current_accuracy)
    except ValueError:
        print("Erreur : current_accuracy n'est pas un float valide.")
        return False

    best_accuracy = 0  # Valeur par défaut si le fichier est vide ou n'existe pas

    # Vérifier si le fichier existe et charger son contenu
    if os.path.exists(metrics_file) and os.path.getsize(metrics_file) > 0:
        with open(metrics_file, "r") as f:
            try:
                best_metrics = json.load(f)
                best_accuracy = float(best_metrics.get("best_accuracy", 0))
                print(f"Meilleure précision précédente : {best_accuracy}")
            except (ValueError, json.JSONDecodeError):
                print(f"Fichier {metrics_file} corrompu. Réinitialisation.")
                best_accuracy = 0

    # Comparer et mettre à jour si nécessaire
    if current_accuracy > best_accuracy:
        print(f"Nouvelle meilleure précision trouvée : {current_accuracy}")
        with open(metrics_file, "w") as f:
            json.dump({"best_accuracy": current_accuracy}, f, indent=4)
        return True

    return False

def update_wandb_tags(project_name, current_run_id, is_best_model):
    """
    Met à jour les tags dans W&B :
    - Supprime `best_model` et `last_evaluation` des anciens runs si nécessaire.
    - Ajoute `last_evaluation` et/ou `best_model` au run actuel.
    - Ajoute un tag `tracked_model` à tous les runs ayant `best_model` ou `last_evaluation`.
    """
    try:
        api = wandb.Api()
        runs = api.runs(project_name)

        for run in runs:
            run_tags = set(run.tags)  # Utilisation de set pour éviter les doublons

            # Si ce n'est pas le run actuel, retirer les anciens tags
            if run.id != current_run_id:
                if "last_evaluation" in run_tags:
                    run_tags.remove("last_evaluation")
                if "best_model" in run_tags and is_best_model:
                    run_tags.remove("best_model")
                
                # Gérer le tag `tracked_model`
                if "best_model" in run_tags or "last_evaluation" in run_tags:
                    run_tags.add("tracked_model")
                else:
                    run_tags.discard("tracked_model")

                # Appliquer les modifications
                run.tags = list(run_tags)
                run.update()

        # Gérer les tags pour le run actuel
        current_run = api.run(f"{project_name}/{current_run_id}")
        current_run_tags = set(current_run.tags)
        current_run_tags.add("tracked_model")
        current_run_tags.add("last_evaluation")
        if is_best_model:
            current_run_tags.add("best_model")

        # Appliquer les modifications au run actuel
        current_run.tags = list(current_run_tags)
        current_run.update()

        print(f"Tags après mise à jour pour le Run ID {current_run_id}: {current_run.tags}")

    except Exception as e:
        print(f"Erreur lors de la mise à jour des tags dans W&B : {str(e)}")



