from fastapi import FastAPI
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from model import PrunableNN

app = FastAPI()

# Load model
model = PrunableNN()
model.load_state_dict(torch.load("results/model_lambda_0.01.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


@app.get("/")
def home():
    return {"message": "Self-Pruning AI Model API is running"}


@app.post("/predict/")
async def predict(file: bytes):
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return {"prediction": classes[predicted.item()]}