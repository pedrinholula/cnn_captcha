from fastapi import APIRouter, UploadFile, File
from utils import preprocess_image, load_model, predict_captcha, decode_predictions
import cv2

router = APIRouter()

# Carregar o modelo uma vez
model = load_model("./models/production.h5")

@router.post("/")
async def predict(file: UploadFile = File(...)):
    """Receive an image as multipart form and try to predict

    parameters:
        file: multipart file captcha
    return:
        data: information about the prediction

    """
    image = await file.read()
    preprocessed_image = preprocess_image(image)
    prediction = predict_captcha(model, preprocessed_image)
    prediction = decode_predictions(prediction,None)
    print(prediction)
    # return "OK"
    return {"prediction": prediction}