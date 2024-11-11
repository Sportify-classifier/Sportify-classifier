
from train import train_model
from evaluate import evaluate_model
from config import config
import os
import random
import torch
import numpy as np

# Définir la graine aléatoire pour les modules random, numpy et torch
random_seed = config['random_seed']
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# S'assurer que les opérations cudnn sont déterministes
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Spécifier les chemins des données
data_dir = config['data_dir']
test_dir = config['test_dir']

# Sélectionner les classes
all_classes = sorted(os.listdir(data_dir))
num_classes = config['num_classes']
if num_classes and num_classes < len(all_classes):
    selected_classes = random.sample(all_classes, num_classes)
else:
    selected_classes = all_classes

# Entraînement
model = train_model(data_dir=data_dir, selected_classes=selected_classes)

# Évaluation
evaluate_model(model, data_dir=test_dir, selected_classes=selected_classes)