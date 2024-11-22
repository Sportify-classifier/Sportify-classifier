from train import train_model
from evaluate import evaluate_model
from config import config
import os
import random
import torch
import numpy as np
from transformers import logging

# Définir la graine aléatoire pour les modules random, numpy et torch
random_seed = config['random_seed']
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# S'assurer que les opérations cudnn sont déterministes
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Spécifier le chemin des données
data_dir = config['data_dir']

# Sélectionner les classes à inclure
all_classes = sorted(os.listdir(data_dir))
excluded_classes = config.get("excluded_classes", [])
included_classes = [cls for cls in all_classes if cls not in excluded_classes]

num_classes = config['num_classes']
if num_classes and num_classes < len(included_classes):
    selected_classes = random.sample(included_classes, num_classes)
else:
    selected_classes = included_classes

# Désactiver temporairement les logs de Hugging Face
logging.set_verbosity_error()

# Entraînement
print("Entraînement en cours...")
model = train_model(data_dir=data_dir, selected_classes=selected_classes)

# Évaluation
print("Évaluation en cours...")
evaluate_model(model, data_dir=data_dir, selected_classes=selected_classes)
