import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
import mlflow
import mlflow.pytorch
from model import SimpleCNN

# 1. Load Configuration
print("Loading configuration...")
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Configure MLflow (The "Memory")
# We use a local SQLite database so we can use the Model Registry features
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("CIFAR10_Project")

# 3. Start the MLflow Run
# Everything inside this 'with' block is tracked
with mlflow.start_run():
    print("Starting MLflow run...")
    
    # A. Log Hyperparameters (The "Recipe")
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
    global_step = 0
    for epoch in range(config['training']['epochs']):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Log training loss with a unique step
            mlflow.log_metric("training_loss", loss.item(), step=global_step)
            global_step += 1
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {epoch_loss:.4f}")
        
        # Log epoch loss with epoch as step
        mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
# # ...existing code...
#     print("Starting training...")
#     for epoch in range(config['training']['epochs']):
#         running_loss = 0.0
#         for i, (images, labels) in enumerate(train_loader):
#             images, labels = images.to(device), labels.to(device)
            
#             # --- The Three-Step Dance ---
#             optimizer.zero_grad()            # 1. Clear gradients
#             outputs = model(images)          # 2. Forward pass
#             loss = criterion(outputs, labels) # 3. Calculate Loss
#             loss.backward()                  # 4. Backward pass
#             optimizer.step()                 # 5. Update weights
#             # -----------------------------
            
#             # Log loss for every batch so we see a smooth curve
#             mlflow.log_metric("training_loss", loss.item())
            
#             running_loss += loss.item()
            
#         # Print epoch average to terminal
#         epoch_loss = running_loss / len(train_loader)
#         print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {epoch_loss:.4f}")
        
#         # Log epoch accuracy/loss as a separate metric if desired
#         mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)

    # E. Save and Register the Model
    # 1. Save locally (as a backup)
    local_path = config['training']['save_path']
    torch.save(model.state_dict(), local_path)
    print(f"Model saved locally to {local_path}")
    
    # 2. Log to Registry (The "Magic" Step)
    # This uploads the model, dependencies, and registers it as a versioned asset.
    # We provide a sample input so MLflow infers the signature (input/output shape)
    sample_input = torch.randn(1, 3, 32, 32).to(device)
    
    mlflow.pytorch.log_model(
        model, 
        "model", 
        registered_model_name="CIFAR10_Model",
        input_example=sample_input.cpu().numpy()
    )
    
    print("Model registered as 'CIFAR10_Model' in the MLflow Registry.")