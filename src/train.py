from data_loader import load_images_and_labels
from model import cnn

# Diretório com os Captchas
data_dir = './data/samples'

# 1. Carregar imagens e rótulos
train_images, test_images, train_labels, test_labels = load_images_and_labels(data_dir, img_shape=(200,50))

model = cnn(input_shape=(50,200, 1), num_classes=19)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Treinar o modelo
history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    batch_size=32,
    epochs=50,  # Ajuste conforme necessário
    verbose=1
)

model.save("./models/production.h5")
