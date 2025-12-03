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



##⚡ Getting Started
1. Local Setup
Clone the repository and install dependencies:

git clone [https://github.com/saikrishnaallam/cifar10-mlops.git](https://github.com/saikrishnaallam/cifar10-mlops.git)
cd cifar10-mlops
pip install -r requirements.txt

## 2. Training the Model 🏋️‍♂️
Run the training pipeline. This script will automatically download the CIFAR-10 dataset, preprocess it, train the CNN, and save the artifact (model.pth).

Bash

python src/train.py
Output: You will see the training loss decrease over epochs.

Artifact: A model.pth file will be generated in the root directory.

## 3. Serving with Docker 🐳
We use Docker to package the model and API into a portable container.

Build the Image:

Bash

docker build -t cifar_app .
Run the Container: This command starts the server, maps port 8000, and mounts your local volume so the container can access the trained model.

Bash

docker run -p 8000:8000 -v $(pwd):/app cifar_app
The API is now live at http://0.0.0.0:8000.

## 4. Testing the Deployment 🧪
To verify the system, run the test script. It downloads a random test image and sends it to your running Docker container for prediction.

Bash

python test_api.py
Expected Output:

Plaintext

Selected Image Index: 193
Actual Label: car
Model Prediction: car
Confidence: 23.86%
✅ Success! The model got it right.
🛠️ API Endpoints
GET /: Health check. Returns {"message": "Welcome to the CIFAR-10 Classifier API!"}.

POST /predict: Accepts an image file and returns the predicted class and confidence score.

🔜 Next Steps (Roadmap)
[ ] Experiment Tracking: Integrate MLflow or Weights & Biases.

[ ] CI/CD: Add GitHub Actions for automated testing.

[ ] Model Registry: Version control for model artifacts.


***

### 💡 Pro Tip for Github
After you save this file, run these commands to update your GitHub repo with this beautiful documentation:

```bash
git add README.md
git commit -m "Add project documentation"
git push

