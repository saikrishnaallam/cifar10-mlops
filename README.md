# 🚀 End-to-End MLOps: CIFAR-10 Image Classification

A production-ready Machine Learning pipeline that demonstrates best practices in **MLOps**. This project goes beyond simple model training by implementing reproducibility, experiment tracking, model versioning, containerization, and automated testing.

---

## 🏗️ Architecture & Tech Stack

| Component | Tool | Description |
| :--- | :--- | :--- |
| **Model** | `PyTorch` | CNN architecture (`Conv2d`, `MaxPool2d`) for image classification. |
| **Serving** | `FastAPI` | Asynchronous REST API to serve predictions. |
| **Container** | `Docker` | Fully containerized environment for reproducibility. |
| **Tracking** | `MLflow` | Tracks experiments (loss/accuracy) and logs parameters. |
| **Registry** | `MLflow Registry` | Version controls models (v1, v2) using a SQLite backend. |
| **CI/CD** | `GitHub Actions` | Automated pipeline to build and test the app on every push. |
| **Config** | `YAML` | Decouples hyperparameters from the source code. |

---

## 📂 Project Structure

```text
cifar10-mlops/
├── .github/workflows/
│   └── ci_pipeline.yaml  # GitHub Actions workflow for automated CI/CD
├── configs/
│   └── config.yaml       # Centralized configuration (LR, Batch Size, Epochs)
├── data/
│   ├── raw/              # Immutable source data (auto-downloaded)
│   └── processed/        # (Optional) Transformed data
├── src/
│   ├── model.py          # PyTorch CNN Architecture class
│   ├── train.py          # Training loop + MLflow Tracking + Registry logic
│   └── main.py           # FastAPI application with Registry fallback logic
├── Dockerfile            # Blueprint for the production API container
├── requirements.txt      # Python dependencies
├── test_api.py           # Client script to simulate user requests/testing
└── README.md             # Project documentation