import sys
from utils import prepare_data, log_class_summary_to_wandb, log_excluded_classes_to_wandb
import wandb
import yaml
import os

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_loader.py <input_data_dir> <output_data_dir>")
        sys.exit(1)

    input_data_dir = sys.argv[1]
    output_data_dir = sys.argv[2]

    # Appeler la fonction pour préparer les données
    prepare_data(input_data_dir, output_data_dir)

    # Charger les paramètres depuis params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Initialiser W&B
    run = wandb.init(
        project="sports-classification",
        resume="never",
        config={
            "learning_rate": params['train']['lr'],
            "batch_size": params['train']['batch_size'],
            "epochs": params['train']['epochs'],
            "num_classes": params['train']['num_classes'],
            "num_folds": params['train']['num_folds'],
            "model_name": params['model']['name'],
            "excluded_classes": params["prepare"].get("excluded_classes", []),
            "max_images_per_class": params["prepare"].get("max_images_per_class", None),
            "split_ratio": params["prepare"].get("split"),
            "sample_fraction": params["prepare"].get("sample_fraction"),
        }
    )

    # Sauvegarder l'identifiant du run pour les étapes suivantes
    with open("wandb_run_id.txt", "w") as f:
        f.write(run.id)

    # Log le tableau récapitulatif des classes
    train_dir = os.path.join(output_data_dir, "train")
    test_dir = os.path.join(output_data_dir, "test")
    log_class_summary_to_wandb(train_dir=train_dir, test_dir=test_dir)

    # Log le tableau récapitulatif des classes exclues
    excluded_classes = params["prepare"].get("excluded_classes", [])
    log_excluded_classes_to_wandb(input_data_dir=input_data_dir, excluded_classes=excluded_classes)
    