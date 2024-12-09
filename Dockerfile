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
    "private_key_id": "b799282cd875c1ac11fe5dab0a498efeeaf111ca", \
    "private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCPhPBaijEOdRmi\\nYkuT8sJZWZ8Dq9FoXUpssxblSZdfmUtHiktIvycoYlm78I3Lrz/j9LSGiCyfHAaD\\n0sVXQm0T87LCe0AszotWt7E3OJM2AgvR+iJhnMshXpJHRLmfqjntwKrsBDY2QuzL\\nnCGQhE+nPplWm3RDLbGcAc6GPDvLIKDuo2BRWwiBWOF9zPOswDeZabQ2lqpMpbkK\\na06RGUbuLqTI716eNjlVMyzwTM7Fmpq0k2dntgzDeUqLv5AyY1G4PqHA0H2ZycQr\\nUoyjniUHQk3LeGm/6b6I0G5Um9HDPKhHIY4jvysMwug42YnNv6q/i5FCtEwZL9R7\\nKUdCd9c1AgMBAAECggEAEr7aZfkGz0ycPIl6NajPmbwK+P/IKkmFHw2FQA2Mp//t\\nxTpa+vV8t9mgvtt8qc4KGkwsCCZhLacLcsxk2jDi7mg2QRC4ISIZMQptKzlIgRVj\\nVdnGA9nm7kTb9lMTq7cgOd2gdZNXQV8cbrvF5ophnQagIJ/NX7joKehSgVX8uVsn\\nhlctQ40D27OS8e3Z3oNFBqn/63b/0HXubR6wFPeSydLCewe1Avv6jLbsJFeCAia0\\n4GeqvsnGGu/mtQ4gdHzbspTROeWVpOusQEkelN5QsjQBegTh3s0TMoSUCW32tEyP\\n//cqvSdHlkbVlc8MyAJQILIza9l0GFKTisVl8b8lQQKBgQDJ2K/ftldqOLs5NgZP\\nCGkTe1iMY5vkePx+g4wVWeZrDv7Vu+3wDmVj5O/JG2cxROGWg+ASi1b5m0hiiZAu\\n5ahrzifzSv6mv11Pid3ueKMko1KEfDFelWfUx+01J413/jCEWrtv5u0ZJFutqWDc\\nOeobW8nMU7ZQMShiNAD9Im3bAwKBgQC2BjMJIxSfdV9+AeP1I7VM+MXVW5//gLU5\\neoRkK/bQWI+eybXTr14Q/dtKIk43xV3j6vE0ISQSiaBZRBL7x07AIRnqyk+zITP0\\n/aV1jh+uEnND3a6GCBYwlZbx8Dnj/WnMdf5w48IjNQLd1LdI4WEKRn+czCedlnvY\\nwd2fFdaTZwKBgERS/PWVG3HxUi7Dgs8t4aOelRKwhQyJh66riLeRftWudcSToK8p\\nsFhoHmLMy9l5n4L7kNW2JPLJBW+VcSDuzFvxpMROFnULQeKyoFUgsNiuEDiYcX26\\nxTLZmgnsIY8ElBe5PslaOdfQ3teiBg+F6yDnqR9pFsV+XlUflVYaWWVRAoGBAIEM\\n5CIWPQjQrmMn8/ZY2rE3rwsVato65kFaG4LpqJMONsTdPYxXSNnDITXuHIZt56Mv\\nbtMGrAx4hrbDDLJ1G+Abl8ReqyLU54FKU4SEvvErI416HcHo+dJ4PAxLxL9fayMK\\nhNqEn59WdjHQHiINqD8gvFjuZSfVCPkkeXvQf9EBAoGBAJIR040GKzg8M63wwbQP\\nu6g6dKPZcoyG2A9hbrwgvNYWV9uLNv6kqXrGPTujOT9qL8xrMmKb0dbce01PdxOc\\n4ICucxpHUXcngAY2FSkkzbTEJ41BySzzWzFB+bbJ9/N0/GCvC3Kug1lLTkgqi+3A\\nIaVqPaUFqzIj+lJQxcp3oBZf\\n-----END PRIVATE KEY-----\\n", \
    "client_email": "sportify-classier@modern-bond-303506.iam.gserviceaccount.com", \
    "client_id": "116568449157343240637", \
    "auth_uri": "https://accounts.google.com/o/oauth2/auth", \
    "token_uri": "https://oauth2.googleapis.com/token", \
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs", \
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/sportify-classier@modern-bond-303506.iam.gserviceaccount.com", \
    "universe_domain": "googleapis.com" \
}' > /app/service-account-key.json

# Vérifier la présence des fichiers et de `.git`
RUN ls -la /app
RUN git status || echo "Git repository not found (expected in CI/CD environments without git)"

# Définir la variable d'environnement pour Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Activer le compte de service
RUN gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"

# Vérifier l'accès à GCS
RUN gsutil ls gs://sportify_classifier || echo "GCS access failed, ensure credentials and permissions are correct"

# Définir le comportement par défaut
CMD ["bash", "-c", "echo 'Running pipeline...' && pwd && ls -la && if [ ! -d .dvc ]; then dvc init --no-scm; fi && dvc repro --pull && echo 'Pipeline finished.' && bash"]