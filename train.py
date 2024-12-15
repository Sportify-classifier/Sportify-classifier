import subprocess
import sys
import torch
import yaml
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from torch.utils.data import DataLoader, Subset
from utils import SportsDataset, create_versioned_dir, calculate_accuracy, log_metrics_to_wandb
import wandb
import bentoml
from PIL.Image import Image
import torch.nn.functional as F


# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


def train_model(data_dir, model_output_dir):
    model_name = params['model']['name']
    num_folds = params['train']['num_folds']
    all_losses_train = []
    all_losses_val = []

    # Read or generate W&B run ID
    run_id_file = "wandb_run_id.txt"
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
    else:
        run_id = wandb.util.generate_id()
        with open(run_id_file, "w") as f:
            f.write(run_id)

    # Join or resume W&B run
    wandb.init(
        project="sports-classification",
        id=run_id,
        resume="allow"
    )

    # Initialize feature extractor
    feature_extractor = EfficientNetImageProcessor.from_pretrained(
        model_name,
        size=params['prepare']['image_size']
    )

    # Load dataset
    dataset = SportsDataset(data_dir=data_dir, feature_extractor=feature_extractor)
    dataset_size = len(dataset)
    fold_size = dataset_size // num_folds
    indices = np.arange(dataset_size)

    for fold in range(1, num_folds + 1):
        print(f"Training fold {fold}/{num_folds}")

        # Split indices for train and validation
        val_indices = indices[(fold - 1) * fold_size:fold * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        # Create DataLoaders
        train_dataloader = DataLoader(train_subset, batch_size=params['train']['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=params['train']['batch_size'], shuffle=False)

        # Initialize model
        model = EfficientNetForImageClassification.from_pretrained(
            model_name,
            num_labels=len(dataset.classes),
            ignore_mismatched_sizes=True
        )
        optimizer = Adam(model.parameters(), lr=params['train']['lr'])
        model.train()

        # Initialize epoch loss lists for this fold
        losses_train = []
        losses_val = []

        for epoch in range(params['train']['epochs']):
            # Training loop
            model.train()
            epoch_loss_train = 0
            epoch_accuracy_train = 0
            for batch in tqdm(train_dataloader):
                inputs = batch['pixel_values'].squeeze(1)
                labels = batch['labels']

                outputs = model(pixel_values=inputs, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss_train += loss.item()
                epoch_accuracy_train += calculate_accuracy(outputs.logits, labels)

            # Average metrics for this epoch
            train_loss = epoch_loss_train / len(train_dataloader)
            train_accuracy = epoch_accuracy_train / len(train_dataloader)

            # Validation loop
            model.eval()
            epoch_loss_val = 0
            epoch_accuracy_val = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = batch['pixel_values'].squeeze(1)
                    labels = batch['labels']

                    outputs = model(pixel_values=inputs, labels=labels)
                    loss = outputs.loss
                    epoch_loss_val += loss.item()
                    epoch_accuracy_val += calculate_accuracy(outputs.logits, labels)

            # Average metrics for this epoch
            val_loss = epoch_loss_val / len(val_dataloader)
            val_accuracy = epoch_accuracy_val / len(val_dataloader)

            # Append losses to fold lists
            losses_train.append(train_loss)
            losses_val.append(val_loss)

            # Log metrics to W&B
            log_metrics_to_wandb(epoch, train_loss, val_loss, train_accuracy, val_accuracy)

        # Append fold losses to global lists
        all_losses_train.append(losses_train)
        all_losses_val.append(losses_val)

    # Calculate mean training and validation losses across folds
    mean_losses_train = np.mean(all_losses_train, axis=0)
    mean_losses_val = np.mean(all_losses_val, axis=0)

    # Plot losses
    epochs = range(1, params['train']['epochs'] + 1)
    plt.plot(epochs, mean_losses_train, label="Training Loss")
    plt.plot(epochs, mean_losses_val, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Average Loss per Epoch")
    plt.legend()

    # Save loss plot
    loss_plot_path = os.path.join(model_output_dir, 'loss_curve.png')
    os.makedirs(model_output_dir, exist_ok=True)
    plt.savefig(loss_plot_path)
    plt.close()

    # Save model locally and with BentoML
    model.save_pretrained(model_output_dir)
    feature_extractor.save_pretrained(model_output_dir)

    # Convertir les classes en python list
    labels_list = list(dataset.classes)

    def preprocess(pil_img: Image):
        pil_img = pil_img.convert('RGB')
        pil_img = pil_img.resize((224, 224))
        x = np.array(pil_img).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # [3,224,224]
        x = np.expand_dims(x, axis=0)  # [1,3,224,224]
        return torch.from_numpy(x)

    def postprocess(model_output):
        # model_output est un "ImageClassifierOutputWithNoAttention"
        logits_tensor = model_output.logits
        probabilities = F.softmax(logits_tensor, dim=1).detach().cpu().numpy().squeeze()
        predicted_class_idx = np.argmax(probabilities)
        return {
            "prediction": labels_list[predicted_class_idx],
            "probabilities": {
                labels_list[i]: float(prob) for i, prob in enumerate(probabilities)
            }
        }
    bentoml.pytorch.save_model(
        name="sports_classifier_model",
        model=model,
        custom_objects={
            "preprocess": preprocess,
            "postprocess": postprocess,
            "labels_list": labels_list,
        },
    )

    bentoml.models.export_model(
        "sports_classifier_model:latest",
        os.path.join(model_output_dir, "sports_classifier_model.bentomodel"),
    )

    print("Model saved to BentoML store and exported!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <data_prepared_dir> <model_output_base_dir>")
        sys.exit(1)

    data_prepared_dir = sys.argv[1]
    model_output_base_dir = sys.argv[2]

    # Set random seeds for reproducibility
    random_seed = params['train'].get("seed", 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    data_dir = os.path.join(data_prepared_dir, 'train')

    # Create a versioned directory for the model
    model_output_dir = create_versioned_dir(model_output_base_dir, "model")

    # Train the model
    train_model(data_dir=data_dir, model_output_dir=model_output_dir)
