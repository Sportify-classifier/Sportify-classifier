FROM python:3.11-slim

# Installer git et autres dépendances éventuelles
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc[gcs]

# Copier tout le code de l'application dans le conteneur
COPY . .

# Renommer dynamiquement le fichier `gha-creds-*.json` en `gha-creds.json`
RUN mv /app/gha-creds-*.json /app/gha-creds.json || echo "GHA credentials file not found, skipping rename"

# Vérifier la présence des fichiers et de `.git`
RUN ls -la /app
RUN git status || echo "Git repository not found (expected in CI/CD environments without git)"

# Définir la variable d'environnement pour Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/gha-creds.json"

# Par défaut, lancer `dvc repro --pull` à chaque démarrage du conteneur
CMD ["bash", "-c", "echo 'Current directory:' && pwd && echo 'Listing files:' && ls -la && echo 'Testing GCS access...' && gsutil ls gs://sportify_classifier && echo 'Running DVC...' && dvc repro --pull 2>&1 | tee dvc_logs.txt && echo 'Pipeline finished.' && bash"]