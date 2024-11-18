from src.model import cnn
from src.data_loader import load_images_and_labels

def model_training(epochs=30):
    # Diretório com os Captchas
    data_dir = './data/samples'

    # 1. Carregar imagens e rótulos
    train_images, test_images, train_labels, test_labels = load_images_and_labels(data_dir, img_shape=(200,50))

    #2. Definir o modelo
    model = cnn(input_shape=(50,200, 1), num_classes=19)

    #3. Compilar o modelo
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    #4 Treinar o modelo
    history = model.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        batch_size=32,
        epochs=epochs,  # Ajuste conforme necessário
        verbose=1
    )

    #5. Salvar o modelo
    model.save("./models/production.h5")
