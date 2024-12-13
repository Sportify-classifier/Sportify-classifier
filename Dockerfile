FROM python:3.11-slim

# Variable
ARG _GIT_KEY
ARG _WANDB_KEY
ARG _SERVICE_ACCOUNT_KEY

ENV GIT_KEY=${_GIT_KEY}
ENV WANDB_KEY=${_WANDB_KEY}
ENV SERVICE_ACCOUNT_KEY=${_SERVICE_ACCOUNT_KEY}

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg \
    jq \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/google-cloud-archive.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/google-cloud-archive.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
    | tee /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && apt-get install -y google-cloud-sdk && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy files
RUN git clone --branch dev https://${GIT_KEY}@github.com/Sportify-classifier/Sportify-classifier.git /app

RUN pip install --no-cache-dir -r requirements.txt

# Set up credentials
RUN echo "${_SERVICE_ACCOUNT_KEY}" | base64 -d > /app/service-account-key2.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key2.json"

RUN gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
RUN gsutil ls gs://sportify_classifier || echo "GCS access failed, ensure credentials and permissions are correct"

RUN wandb login ${WANDB_KEY}

CMD ["bash", "-c", "echo 'Running pipeline...' && pwd && ls -la && dvc repro --pull && echo 'Pipeline finished.' && bash"]