# **Sportify-Classifier**  
**Sports Image Recognition with Continuous Improvement MLOps Pipeline**

---

## **Team**
- Komivi ASSIAMUA
- Veton BERISHA
- Raphaël BESSON
- Assagnon KOUMA
- Maxime PERROD

---

## **Project Summary**
This project aims to deploy and continuously improve a sports image recognition model using an MLOps pipeline. By regularly integrating new data, we enhance model accuracy and robustness. Key metrics are monitored in real-time to track performance and detect model drift.

---

## **Pipeline Overview**
1. **Base Model**: Using [EfficientNet-B0](https://huggingface.co/google/efficientnet-b0) from Hugging Face.  
2. **Git & DVC Setup**: Data versioning with DVC.  
3. **CI/CD on Google Cloud**: Automatic fine-tuning triggered on push.  
4. **Result Analysis**: Pull request generated with metrics and performance graphs.  
5. **Deployment**: Model deployed on Google Cloud (Terraform) for online testing.  
6. **Monitoring**: Performance tracked with Grafana or Bayesian optimization tools.

---

## **Resources**
- **GitHub**: [Sportify Classifier Repository](https://github.com/Sportify-classifier/Sportify-classifier)  
- **Dataset**: [Sports Classification Dataset on Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification)  
- **Model**: [EfficientNet-B0 on Hugging Face](https://huggingface.co/google/efficientnet-b0)

---

## ** Warning **
On a eu des problèmes avec google vertex AI pour l'entrainement de notre modèle. L'entrainement fonctionnait mais pour une raison inconnue, google vertex ne voulait plus accepté la commande "dvc pull". Nous avons donc décidé de passer par le github runner à la dernière minute pour rendre quelque chose de fonctionelle.


## **Setup Instructions**

### **1. Create and Activate a Virtual Environment**
Run the following commands to set up and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac
.venv\Scripts\activate     # On Windows
```

### **2. Install Dependencies**
Install the required dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```
### **3. Configure Google Cloud Access**
1. Set up the required environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="service-account-key.json"
```
2. Verify if gcloud CLI is Installed

Check if `gcloud` is installed by running:
```bash
gcloud --version
```
If `gcloud` is not installed, follow the installation instructions in this link : https://cloud.google.com/sdk/docs/install

3. Authenticate the service account

Activate the service account with the following command:
```bash
gcloud auth activate-service-account --key-file="service-account-key.json"
```
This step ensures the `gcloud` CLI uses the credentials in the service account key file. We can verify if the account is correctly activated with the following command:
```bash
gcloud auth list
```
Check if the service account is active. If not, explicitly set it as the active account with the following command:


6. Verify the connection to Google Cloud by listing the project and buckets:
```bash
gcloud config set project modern-bond-303506
gcloud storage buckets list
```

### **4. Download the Dataset**
Download the dataset from Google Cloud Storage using DVC:
```bash
dvc pull
```
If everything is set up correctly, this will download the dataset into the appropriate directories (e.g., data/all_data).

### **5. Adjust the params.yaml**
Modify the params.yaml to adjust the training parameters. 

### **6. Train the Model**
Commit and push the changes to the repository. It will execute the CI/CD pipeline and train the model with the new parameters.
If you want to do it locally, you need to run the command
```bash
dvc repro
```
don't forget to do
```bash
dvc push
```