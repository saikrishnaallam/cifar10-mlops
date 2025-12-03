import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
from model import SimpleCNN

# 1. Load Configuration
print("Loading configuration...")
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Prepare Data (using values from config)
transform = transforms.Compose([
    transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(config['data']['normalization_mean'], config['data']['normalization_std'])
])

print("Downloading dataset...")
train_dataset = datasets.CIFAR10(root='./data/raw', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)

# 3. Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config['training']['learning_rate'])

# 4. Training Loop
print("Starting training...")
for epoch in range(config['training']['epochs']):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # --- The Three-Step Dance ---
        
        # 1. Zero the gradients (Clear the previous "blame")
        optimizer.zero_grad()
        
        # 2. Forward pass & Loss calculation (The "Judge")
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 3. Backward pass & Step (The "Mechanic")
        loss.backward()
        optimizer.step()
        
        # -----------------------------
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {running_loss / len(train_loader):.4f}")

# 5. Save the artifact
torch.save(model.state_dict(), config['training']['save_path'])
print(f"Model saved to {config['training']['save_path']}")