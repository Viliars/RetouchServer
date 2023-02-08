import torch
import torchvision
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, status, HTTPException, File, UploadFile
from fastapi.responses import Response
from pathlib import Path
import os
from loguru import logger
from utils import StartupException
import warnings
from time import time
warnings.filterwarnings("ignore")

app = FastAPI()

model = None

ToTensor = torchvision.transforms.ToTensor()
ToPILImage = torchvision.transforms.ToPILImage()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    logger.warning("CUDA is not available. CPU will be used")
    device = torch.device("cpu")


@app.on_event("startup")
def start_service():
    logger.info("Service is starting")

    model_path = os.getenv("MODEL_PATH")
    if model_path is None:
        logger.error("ENV variable MODEL_PATH does not exist")
        raise StartupException("ENV variable MODEL_PATH does not exist")

    model_path = Path(model_path)

    if not model_path.exists():
        logger.error("Model does not exist")
        raise StartupException("Model does not exist")

    global model
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    logger.info("Model is loaded")

    logger.info("Preparations for the launch are over")


@app.post("/retouch")
def retouch(image: bytes = File()):
    image = Image.open(BytesIO(image))
    tensor = ToTensor(image)
    tensor = tensor.unsqueeze(0).to(device)

    start = time()
    with torch.no_grad():
        result = model(tensor)
    end = time()

    print(end - start)

    result = result.squeeze(0)
    result_image = ToPILImage(result)

    print(result_image.size)

    img_byte_arr = BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    image_bytes = img_byte_arr.getvalue()

    return Response(content=image_bytes, media_type="image/png")



@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    global model
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"status": "ok"}