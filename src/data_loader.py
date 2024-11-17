import os
import numpy as np
import cv2
from image_processing import image_processing

def load_images_and_labels(data_dir, img_shape=(50, 200)):
    images = []
    labels = []
    # dilation or close
    method = "dilation"

    # Listar todos os arquivos na pasta
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            # Caminho completo para a imagem
            img_path = os.path.join(data_dir, file_name)
            img = image_processing(img_path,
                    gaussian_kernel=None, sigma=0.5,
                    median_kernel=None,
                    closing_k=(5,5),
                    dilation_k=(3,5),
                    method = method)
            # Carregar a imagem e redimensionar
            img = cv2.resize(img, img_shape)
            images.append(img)

            # Obter o rótulo (nome do arquivo sem a extensão)
            label = os.path.splitext(file_name)[0]
            labels.append(label)

    # Converter listas para arrays numpy
    images = np.array(images, dtype=np.float32) / 255.0  # Normalizar pixels (0 a 1)
    labels = np.array(labels)
    # return images, labels

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    train_labels, train_encoder = encode_labels(train_labels)
    test_labels, test_encoder = encode_labels(test_labels)

    return train_images, test_images, train_labels, test_labels