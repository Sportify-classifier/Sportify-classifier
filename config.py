config = {
    "num_epochs": 1,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "sample_fraction": 0.4,  # Pour l'entraînement 40 % de la data train
    "test_sample_fraction": 1,  # Pour le test 100 % de la data test
    "num_classes": 20,  # Nombre de classes à utiliser
    "random_seed": 42,
    "model_name": "google/efficientnet-b0",
    "data_dir": "./data/train", # A modifier si besoin
    "test_dir": "./data/test"  # A modifier si besoin
     
}