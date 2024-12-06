FROM python:3.11-slim

# Installer git et autres dépendances éventuelles
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installer DVC avec support GCS
RUN pip install dvc[gcs]

# Copier tout le code
COPY . .

# Par défaut, le conteneur lance un shell (vous spécifierez la commande lors du job Vertex)
CMD ["bash"]