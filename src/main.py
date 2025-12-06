import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
import mlflow.pytorch
from model import SimpleCNN

app = FastAPI()

# --- Configuration ---
MODEL_NAME = "CIFAR10_Model"
STAGE = "Production"  # We only want to serve the model marked as Production

# --- Model Loading Logic ---
print("Initializing API...")

model = None

try:
    # Attempt 1: Load from MLflow Registry
    # This ensures we are always serving the version approved for Production
    print(f"Attempting to load model '{MODEL_NAME}' (Stage: {STAGE}) from Registry...")
    
    # Point to the local database where we stored the registry
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Load the model
    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{STAGE}")
    print("✅ Successfully loaded model from MLflow Registry.")

except Exception as e:
    print(f"⚠️ Could not load from Registry: {e}")
    print("Attempting fallback to local 'model.pth'...")
    
    # Attempt 2: Fallback to Local File
    # Useful for local testing when the database might not exist or be accessible
    try:
        model = SimpleCNN()
        model.load_state_dict(torch.load("model.pth", map_location='cpu'))
        print("✅ Successfully loaded model from local file.")
    except FileNotFoundError:
        print("❌ CRITICAL: No model found in registry or locally. API will fail on prediction.")

# Switch to evaluation mode (freezes layers like Dropout)
if model:
    model.eval()

# --- Preprocessing ---
# MUST match the transformation used during training
inference_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# CIFAR-10 Class Names
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the CIFAR-10 Classifier API!",
        "model_source": "Registry" if "mlflow" in str(type(model)) else "Local File"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded."}

    # 1. Read and Transform Image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Handle non-RGB images (like PNGs with transparency)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    input_tensor = inference_transforms(image).unsqueeze(0) # Add batch dim (1, 3, 32, 32)
    
    # 2. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # 3. Probabilities
        probs = F.softmax(outputs, dim=1)
        
    # 4. Result
    confidence, predicted_class_index = torch.max(probs, 1)
    predicted_label = classes[predicted_class_index.item()]
    
    return {
        "prediction": predicted_label,
        "confidence": f"{confidence.item() * 100:.2f}%"
    }