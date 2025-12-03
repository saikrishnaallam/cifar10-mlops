# 🚀 CIFAR-10 End-to-End MLOps Project

A robust, production-ready Machine Learning pipeline for image classification. This project demonstrates best practices in MLOps, moving beyond simple notebooks to a modular, containerized, and scalable system.

## 🏗️ Architecture & Technologies

* **PyTorch**: Custom CNN architecture (`SimpleCNN`) for classifying images into 10 categories.
* **FastAPI**: High-performance asynchronous REST API for model serving.
* **Docker**: Fully containerized environment ensuring reproducibility across machines.
* **Hydra/YAML**: Configuration management to decouple hyperparameters from code.
* **Project Structure**: Clean separation of data, source code, and configurations.

## 📂 Project Structure

```text
cifar10-mlops/
├── configs/
│   └── config.yaml       # Hyperparameters (learning rate, batch size, etc.)
├── data/
│   ├── raw/              # Immutable source data (auto-downloaded)
│   └── processed/        # Transformed data for training
├── src/
│   ├── model.py          # PyTorch Model Architecture (CNN)
│   ├── train.py          # Training loop with modular "Three-Step Dance"
│   └── main.py           # FastAPI inference endpoint
├── Dockerfile            # Blueprint for the production container
├── requirements.txt      # Python dependencies
├── test_api.py           # Client script to test the deployed API
└── README.md             # Project documentation
