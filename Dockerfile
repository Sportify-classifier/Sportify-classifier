FROM python:3.11-slim

ARG _GIT_KEY
ARG _WANDB_KEY
ARG _SERVICE_ACCOUNT_KEY
ENV GIT_KEY=${_GIT_KEY}
ENV WANDB_KEY=${_WANDB_KEY}
ENV SERVICE_ACCOUNT_KEY=${_SERVICE_ACCOUNT_KEY}

# Installer les dépendances nécessaires
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Ajouter la clé GPG de Google Cloud
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/google-cloud-archive.gpg

# Ajouter le dépôt APT de Google Cloud
RUN echo "deb [signed-by=/usr/share/keyrings/google-cloud-archive.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list

# Installer le SDK Google Cloud
RUN apt-get update && apt-get install -y google-cloud-sdk && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Cloner le dépôt GitHub en utilisant le Personal Access Token
RUN echo "GIT_KEY is: ${GIT_KEY}"
RUN git clone --branch dev https://${GIT_KEY}@github.com/Sportify-classifier/Sportify-classifier.git /app

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Décoder la clé de service à partir de base64 et la sauvegarder dans un fichier
RUN echo "${_SERVICE_ACCOUNT_KEY}" | base64 -d > /app/service-account-key2.json

# Vérifier le fichier JSON décodé
RUN cat /app/service-account-key2.json | jq .

# Définir la variable d'environnement pour Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key2.json"

# Activer le compte de service
RUN gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"

# Vérifier l'accès à GCS
RUN gsutil ls gs://sportify_classifier || echo "GCS access failed, ensure credentials and permissions are correct"

# Log to Weights & Biases
RUN wandb login ${WANDB_KEY}

# Définir le comportement par défaut
CMD ["bash", "-c", "echo 'Running pipeline...' && pwd && ls -la && dvc repro --pull && echo 'Pipeline finished.' && bash"]