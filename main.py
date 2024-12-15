from train import train_model
from evaluate import evaluate_model
import os
import random
import yaml
import torch
import numpy as np

#PLUS BESOIN D'UTILISER CE FICHIER APRES AVOIR CREER LA PIPELINE

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

random_seed = params['train'].get("seed")
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = params['data'].get("dir")

all_classes = sorted(os.listdir(data_dir))
excluded_classes = params['prepare'].get("excluded_classes")
included_classes = [cls for cls in all_classes if cls not in excluded_classes]

num_classes = params['train'].get("num_classes")
if num_classes and num_classes < len(included_classes):
    selected_classes = random.sample(included_classes, num_classes)
else:
    selected_classes = included_classes

# Entraînement
print("Entraînement en cours...")
model = train_model(data_dir=data_dir, selected_classes=selected_classes)

# Évaluation
print("Évaluation en cours...")
evaluate_model(model, data_dir=data_dir, selected_classes=selected_classes)
