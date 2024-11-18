from tensorflow.keras.models import load_model as lm
from PIL import Image
import numpy as np
import cv2

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
    # img = np.expand_dims(img, axis=-1)
    img = img/255.0
    img = np.expand_dims(img, axis=0)
    processed = image_processing(img)
    processed = cv2.resize(processed, (200, 50))
    print('pre', processed.shape)

    return processed

def load_model(path):
    return lm(path)

def predict_captcha(model, image):
    print("captcha",image.shape)
    predictions = model.predict(image)
    predicted_classes = np.argmax(predictions, axis=-1)
    return predicted_classes

def image_processing(
        img,
        gaussian_kernel=None, sigma=0,
        median_kernel=None,
        closing_k=(3,3),
        dilation_k=(3,5),
        method = 'closing'):
    kernel_d = np.ones(dilation_k, np.uint8)
    kernel_c = np.ones(closing_k, np.uint8)

    # (h, w) = img.shape[:2]
    # print(img.shape)
    # img = cv2.resize(img, (int(w*1.8), int(h*1.8)))
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    print("image", thresh.shape)

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