# **Sportify-Classifier**  
**Sports Image Recognition with Continuous Improvement MLOps Pipeline**

---

## **Team**
- Perrod Maxime  
- Assiamua Komivi  
- Berisha Veton  
- Besson Raphaël  
- Kouma Assagnon  

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
- **GitHub**: [Sportify Classifier Repository](https://github.com/TWAAXOne/Sportify-Classifier)  
- **Dataset**: [Sports Classification Dataset on Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification)  
- **Model**: [EfficientNet-B0 on Hugging Face](https://huggingface.co/google/efficientnet-b0)

---

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
To connect to the dataset stored on Google Cloud Storage, follow these steps:
1. Download the file service-account-key.json shared on Teams.
2. Place the file in the root of this repository:
```bash
project-folder/
├── dvc.yaml
├── params.yaml
├── service-account-key.json
└── ...
```
3. Set up the required environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="service-account-key.json"
```
4. Verify the connection to Google Cloud by listing the project and buckets:
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
Modify the params.yaml to adjust the training parameters. And push the changes to the repository.
It will execute the CI/CD pipeline and train the model with the new parameters.