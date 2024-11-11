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
