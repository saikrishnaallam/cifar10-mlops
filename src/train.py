import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
import mlflow
from model import SimpleCNN

# 1. Load Configuration
print("Loading configuration...")
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Set the MLflow Experiment Name
# This groups all your runs under one banner in the dashboard
mlflow.set_experiment("CIFAR10_Project")

# 3. Start the MLflow Run
# We wrap the entire process so everything is tracked together
with mlflow.start_run():
    print("Starting MLflow run...")
    
    # A. Log Parameters (The "Recipe")
    # We log the dictionaries directly
    mlflow.log_params(config['training'])
    mlflow.log_params(config['data'])

    # B. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(config['data']['normalization_mean'], config['data']['normalization_std'])
    ])

    print("Downloading dataset...")
    train_dataset = datasets.CIFAR10(root='./data/raw', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)

    # C. Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['training']['learning_rate'])

    # D. Training Loop
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # --- The Three-Step Dance ---
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            # -----------------------------
            
            # E. Log the Metric (The "Score")
            # We log every batch to see the fine-grained progress
            mlflow.log_metric("training_loss", loss.item())
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {running_loss / len(train_loader):.4f}")

    # E. Save and Log the Artifact
    save_path = config['training']['save_path']
    torch.save(model.state_dict(), save_path)
    print(f"Model saved locally to {save_path}")
    
    # Upload the file to MLflow
    mlflow.log_artifact(save_path)
    print("Model artifact uploaded to MLflow.")