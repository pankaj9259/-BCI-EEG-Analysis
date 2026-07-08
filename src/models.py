import tensorflow as tf
from tensorflow.keras import layers, models


def build_simple_ann(input_dim, hidden_units=128):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(hidden_units//2, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
