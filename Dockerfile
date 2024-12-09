FROM python:3.11-slim

# Installer les dépendances nécessaires, y compris git et curl pour installer gcloud
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg && \
    rm -rf /var/lib/apt/lists/*

# Ajouter le dépôt officiel pour gcloud et installer le SDK Google Cloud
RUN curl -sSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc[gcs]

# Copier tout le code de l'application dans le conteneur
COPY . .

# Vérifier la présence des fichiers et de `.git`
RUN ls -la /app
RUN git status || echo "Git repository not found (expected in CI/CD environments without git)"

# Définir la variable d'environnement pour Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Activer le compte de service
RUN gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"

# Par défaut, lancer `dvc repro --pull` à chaque démarrage du conteneur
CMD ["bash", "-c", "echo 'Current directory:' && pwd && echo 'Listing files:' && ls -la && echo 'Testing GCS access...' && gsutil ls gs://sportify_classifier && echo 'Running DVC...' && dvc repro --pull 2>&1 | tee dvc_logs.txt && echo 'Pipeline finished.' && bash"]