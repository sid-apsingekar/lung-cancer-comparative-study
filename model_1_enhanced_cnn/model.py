import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Input
)

from config import *


def build_enhanced_cnn():

    model = Sequential([

        Input(shape=(224, 224, 3)),

        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D(pool_size=(2, 2)),


        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D(pool_size=(2, 2)),


        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        Dropout(0.4),

        Flatten(),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(0.4),

        Dense(
            128,
            activation="relu"
        ),

        Dense(
            NUM_CLASSES,
            activation="softmax"
        )

    ])

    return model