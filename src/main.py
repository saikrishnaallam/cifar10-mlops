import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
from .model import SimpleCNN

app = FastAPI()

# 1. Load the Model Architecture
# We instantiate the class we defined in model.py
model = SimpleCNN()

# 2. Load the Trained Weights
# map_location='cpu' ensures this works on your laptop even if you trained on a GPU
# We use 'try/except' just in case you haven't run train.py yet
try:
    model.load_state_dict(torch.load("model.pth", map_location='cpu'))
    model.eval()  # Important: Switch to evaluation mode (freezes layers like Dropout)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Warning: model.pth not found. Please run train.py first.")

# 3. Define the Preprocessing (MUST match training data)
inference_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# CIFAR-10 Class Names
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.get("/")
def read_root():
    return {"message": "Welcome to the CIFAR-10 Classifier API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # A. Read and transform the image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Ensure image is RGB (converts grayscale or PNG alpha channels)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Add batch dimension (1, 3, 32, 32)
    input_tensor = inference_transforms(image).unsqueeze(0)
    
    # B. Run Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # C. Convert to Probabilities
        probs = F.softmax(outputs, dim=1)
        
    # D. Get the winner
    confidence, predicted_class_index = torch.max(probs, 1)
    predicted_label = classes[predicted_class_index.item()]
    
    return {
        "prediction": predicted_label,
        "confidence": f"{confidence.item() * 100:.2f}%"
    }