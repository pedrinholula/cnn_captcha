from tensorflow.keras.models import load_model as lm
from PIL import Image
import numpy as np
import cv2
import pickle

def preprocess_image(image):
    """
    Receive a bytecode image and decode to a CV image with preprocessing
    parameters:
        image: imagebuffer
    return:
        image: preprocessed image
    """
    img = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=-1)
    img = img/255.0
    # img = np.expand_dims(img, axis=0)
    processed = image_processing(img)
    processed = cv2.resize(processed, (200, 50))
    return processed

def load_model(path):
    """
    Load the model
    parameters:
        path: Path of the model;
    return:
        tf loaded model
    """
    return lm(path)

def predict_captcha(model, image):
    """
    Predict the image using the model
    parameters:
        model: Model to predict
        image: image to be predicted
    return:
        Classes predicted
    """
    img = np.expand_dims(image, axis=-1)  # Adiciona dimensão do canal: (1, 50, 200, 1)
    img = np.expand_dims(img, axis=0)  # Adiciona dimensão do lote: (1, 50, 200)
    predictions = model.predict(img)
    predicted_classes = np.argmax(predictions, axis=-1)
    # return 1
    return predicted_classes

def image_processing(
        img,
        gaussian_kernel=None, sigma=0,
        median_kernel=None,
        closing_k=(3,5),
        dilation_k=(3,5),
        method = 'closing'):
    """
    image pre-processing to use on model.
    parameters:
        img: image to be pre-processed
        gaussian_kernel: Kernel to use on gaussian filter or None to pass,
        sigma: Gaussian filter sigma,
        median_kernel: Kernel to use on median filter or None to pass,
        closing_k: Kernel to use on closing filter. Default (3,3),
        dilation_k: Kernel to use on dilation filter. Default (3,5),
        method = Method od filtering - closing, dilation or None. Default 'closing'.
    """
    kernel_d = np.ones(dilation_k, np.uint8)
    kernel_c = np.ones(closing_k, np.uint8)

    (h, w) = img.shape[:2]
    img = cv2.resize(img, (int(w*1.8), int(h*1.8)))
    ret, thresh = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY)

    if median_kernel != None:
        thresh = cv2.medianBlur(thresh, median_kernel)

    if gaussian_kernel != None:
        thresh = cv2.GaussianBlur(thresh, (gaussian_kernel, gaussian_kernel), sigma)

    if method == 'dilation' and dilation_k != None:
        dilation = cv2.dilate(thresh, kernel_d, iterations=1)
        # dilation_image = Image.fromarray(dilation, mode="L")
        # dilation_image.save(tmp_path,format='PNG')
        # dilation_buffer = io.BytesIO()
        # dilation_image.save(dilation_buffer,format='PNG')
        return dilation
    elif method == 'closing' and closing_k != None:
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
        # closing_image = Image.fromarray(closing, mode="L")
        # closing_image.save(tmp_path,format='PNG')
        # closing_buffer = io.BytesIO()
        # closing_image.save(closing_buffer,format='PNG')
        return closing
    else:
        return thresh


def decode_predictions(predictions, encoder=None):

    """
    Decode as predictions to string.

    Args:
        predictions: Indexes array.
        encoder: Encoder. None will use default encoder

    Returns:
        list: Decoded strings.
    """

    # Carregar o encoder salvo se não foi passado o encoder
    if encoder == None:
        with open("./models/production_encoder.pkl", "rb") as f:
            enc = pickle.load(f)
    else:
        enc=encoder
    # Redimensiona para passar ao inverse_transform
    # Obter as classes do encoder (os caracteres correspondentes às posições)
    classes = enc.categories_[0]
    print(classes, predictions)

    # Mapear índices para os caracteres
    decoded_string = "".join(classes[index] for index in predictions[0])
    return decoded_string