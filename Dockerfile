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

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc[gcs]

# Copier tout le code de l'application dans le conteneur
COPY . .

# Créer et configurer le fichier service-account-key.json
RUN echo '{ \
  "type": "service_account", \
  "project_id": "modern-bond-303506", \
  "private_key_id": "your_private_key_id", \
  "private_key": "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n", \
  "client_email": "sportify-classier@modern-bond-303506.iam.gserviceaccount.com", \
  "client_id": "your_client_id", \
  "auth_uri": "https://accounts.google.com/o/oauth2/auth", \
  "token_uri": "https://oauth2.googleapis.com/token", \
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs", \
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/sportify-classier%40modern-bond-303506.iam.gserviceaccount.com" \
}' > /app/service-account-key.json

# Définir la variable d'environnement pour Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Activer le compte de service
RUN gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"

# Vérifier l'accès à GCS
RUN gsutil ls gs://sportify_classifier || echo "GCS access failed, ensure credentials and permissions are correct"

# Définir le comportement par défaut
CMD ["bash", "-c", "echo 'Running pipeline...' && dvc repro --pull && echo 'Pipeline finished.' && bash"]