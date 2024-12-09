FROM python:3.11-slim

# Installer les dépendances nécessaires, y compris gnupg pour la gestion des clés GPG
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Ajouter la clé GPG de Google Cloud et la stocker dans /usr/share/keyrings/
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/google-cloud-archive.gpg

# Ajouter le dépôt APT de Google Cloud avec l'attribut signed-by
RUN echo "deb [signed-by=/usr/share/keyrings/google-cloud-archive.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list

# Mettre à jour les listes de paquets et installer le SDK Google Cloud
RUN apt-get update && apt-get install -y google-cloud-sdk && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc[gcs]

# Copier tout le code de l'application dans le conteneur
COPY . .
COPY service-account-key.json /app/service-account-key.json

# Vérifier la présence des fichiers et de `.git`
RUN ls -la /app
RUN git status || echo "Git repository not found (expected in CI/CD environments without git)"

# Définir la variable d'environnement pour Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Activer le compte de service
RUN gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"

# Par défaut, lancer `dvc repro --pull` à chaque démarrage du conteneur
CMD ["bash", "-c", "echo 'Current directory:' && pwd && echo 'Listing files:' && ls -la && echo 'Testing GCS access...' && gsutil ls gs://sportify_classifier && echo 'Running DVC...' && dvc repro --pull 2>&1 | tee dvc_logs.txt && echo 'Pipeline finished.' && bash"]