FROM python:3.11-slim

# Installer git et autres dépendances éventuelles
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc[gcs]

# Copier tout le code
COPY . .

# Vérifier la présence de .git
RUN ls -la
RUN ls -la /app
RUN ls -la /app && git status || echo "Git repository not found"

# Par défaut, lancer `dvc repro --pull` à chaque démarrage du conteneur
CMD ["bash", "-c", "echo 'Current directory:' && pwd && echo 'Listing files:' && ls -la && echo 'Testing GCS access...' && gsutil ls gs://sportify_classifier && echo 'Running DVC...' && dvc repro --pull 2>&1 | tee dvc_logs.txt && echo 'Pipeline finished.' && bash"]