FROM python:3.11-slim

ARG _GIT_KEY
ARG _WANDB_KEY
ARG _SERVICE_ACCOUNT_KEY
ENV GIT_KEY=${_GIT_KEY}
ENV WANDB_KEY=${_WANDB_KEY}
ENV SERVICE_ACCOUNT_KEY=${_SERVICE_ACCOUNT_KEY}

# Installer les dépendances système
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

WORKDIR /app

# Cloner le dépôt avec la clé Git
RUN echo "GIT_KEY is: ${GIT_KEY}"
RUN git clone --branch dev https://${GIT_KEY}@github.com/Sportify-classifier/Sportify-classifier.git /app

# Installer les dépendances Python, incluant dvc[gcs]
RUN pip install --no-cache-dir -r requirements.txt dvc[gcs]

# Installer wandb et se connecter
RUN wandb login ${WANDB_KEY}

# Décoder la clé de service GCP et l'utiliser
RUN echo "${_SERVICE_ACCOUNT_KEY}" | base64 -d > /app/service-account-key2.json
RUN cat /app/service-account-key2.json | jq .

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key2.json"
RUN gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"

# Vérifier l'accès au bucket GCS
RUN gsutil ls gs://sportify_classifier || echo "GCS access failed, ensure credentials and permissions are correct"

# Vérifier le remote
RUN dvc remote list

# Le CMD lance finalement le repro (les données sont déjà en cache grâce à dvc pull)
CMD ["bash", "-c", "echo 'Running pipeline...' && pwd && ls -la && ls data && dvc repro --pull && echo 'Pipeline finished.' && bash"]