config = {
    "num_epochs": 1,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "sample_fraction": 0.4,  # Fraction proportionnelle pour entraînement (validation) et test
    "train_split": 0.7,  # Par défaut, 70% pour l'entraînement
    "num_classes": 20,  # Nombre de classes à utiliser
    "num_folds": 1, # Nombre de dossiers pour la cross-validation (3-5 max), possible de mettre 1 (sans cross-validation) mais pas 0
    "random_seed": 42,
    "model_name": "google/efficientnet-b0",
    "data_dir": "./data/all_data",  # Unique dossier contenant les données (créer après avoir exécuter fusion.py)
    "excluded_classes": ["sky surfing"],  # Exclure la classe "sky surfing" (voir category_counts.txt)
    "max_images_per_class": 107  # Maximum d'images par classe pour équilibrage (voir category_counts.txt)
}
