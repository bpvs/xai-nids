from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten,
                                     Dense, Dropout, BatchNormalization)

def build_1d_cnn(input_shape, num_classes=1):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model