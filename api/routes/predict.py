from fastapi import APIRouter, UploadFile, File
from utils import preprocess_image, load_model, predict_captcha
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
    print("predict",preprocessed_image.shape)
    cv2.imshow('Closing', preprocessed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    prediction = predict_captcha(model, preprocessed_image)
    # return "OK"
    # return {"prediction": prediction}