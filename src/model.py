from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Bidirectional, LSTM, Dense, Input, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Attention

def cnn(input_shape, num_classes):
    timesteps = 5
    # Input
    inputs = Input(shape=input_shape, name="input_layer")

    # Convolutional Layers
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and Reshape for RNN Input
    x = Reshape((timesteps, -1))(x)

    # Bidirectional LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(x)

    # Attention Mechanism
    attention_output = Attention()([x, x])

    # Fully Connected Layer for Classification
    x = Dense(num_classes, activation="softmax", name="output_layer")(attention_output)

    outputs = Reshape((timesteps, num_classes))(x)
    # Model
    model = Model(inputs, outputs)
    return model