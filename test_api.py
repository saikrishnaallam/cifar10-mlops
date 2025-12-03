import requests
import random
import io
from torchvision import datasets

# 1. Define the API Endpoint
# Note: On your local machine, '0.0.0.0' inside Docker maps to 'localhost' or '127.0.0.1'
url = "http://127.0.0.1:8000/predict"

# 2. Load the Test Data (Raw images, no transforms yet)
# We want the raw PIL image to simulate a user uploading a file
print("Loading Test Data...")
test_dataset = datasets.CIFAR10(root='./data/raw', train=False, download=True)

# 3. Pick a Random Image
idx = random.randint(0, len(test_dataset) - 1)
image, label_index = test_dataset[idx]

# Get the class name from the label index
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
actual_label = classes[label_index]

print(f"\nSelected Image Index: {idx}")
print(f"Actual Label: {actual_label}")

# 4. Convert Image to Bytes (to send over HTTP)
# This mimics 'uploading' a file
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

# 5. Send Request to Docker API
try:
    files = {'file': ('test_image.png', img_byte_arr, 'image/png')}
    response = requests.post(url, files=files)
    
    # 6. Print Result
    if response.status_code == 200:
        result = response.json()
        print(f"Model Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
        
        if result['prediction'] == actual_label:
            print("✅ Success! The model got it right.")
        else:
            print("❌ Oops! The model made a mistake.")
    else:
        print(f"Error: {response.status_code} - {response.text}")

except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the server. Is your Docker container running?")