FROM python:3.11-slim

# Installer les dépendances nécessaires
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg \
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
ARG COMMIT_SHA

RUN git clone https://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@github.com/Sportify-classifier/Sportify-classifier.git /app && \
    cd /app && \
    git checkout ${COMMIT_SHA}

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc[gcs]


# Définir la variable d'environnement pour Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Activer le compte de service
RUN gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"

# Vérifier l'accès à GCS
RUN gsutil ls gs://sportify_classifier || echo "GCS access failed, ensure credentials and permissions are correct"

# Définir le comportement par défaut
CMD ["bash", "-c", "echo 'Running pipeline...' && pwd && ls -la && if [ ! -d .dvc ]; then dvc repro --pull && echo 'Pipeline finished.' && bash"]