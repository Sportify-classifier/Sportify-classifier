# Sportify-Classifier (Sports Image Recognition with Continuous Improvement MLOps Pipeline)

## Team

- Perrod Maxime, Assiamua Komivi, Berisha Veton, Besson Raphaël, Kouma Assagnon

## Project Summary

This project aims to deploy and continuously improve a sports image recognition model using an MLOps pipeline. By regularly integrating new data, we enhance model accuracy and robustness. Key metrics are monitored in real-time to track performance and detect model drift.

## Pipeline Overview

1. **Base Model**: Using [EfficientNet-B0](https://huggingface.co/google/efficientnet-b0) from Hugging Face.
2. **Git & DVC Setup**: Data versioning with DVC.
3. **CI/CD on GCloud**: Automatic fine-tuning triggered on push.
4. **Result Analysis**: Pull request generated with metrics and performance graphs.
5. **Deployment**: Model deployed on Google Cloud (Terraform) for online testing.
6. **Monitoring**: Performance tracked with Grafana or Waits on Bayes.

## Resources

- **GitHub**: [Sportify Classifier Repo](https://github.com/TWAAXOne/Sportify-Classifier)
- **Dataset**: [Sports Classification on Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
- **Model**: [EfficientNet-B0 on Hugging Face](https://huggingface.co/google/efficientnet-b0)

# Instructions for testing the model

1. Create and activate a virtual environment
2. Install the dependencies with requirements.txt
3. Upload the data to kaggle and put the data folder in the current repository 
4. Run the data_merging.py file to create a all_data folder with data from train, validation and test kaggle folders
3. Upload the data with this command:
```shell
dvc pull
```
5. Adjust the hyperparameters in the params.yaml file
6. Run the main.py file and close the graph window each time it appears to allow the script to continue until it is fully executed
