import numpy as np
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SpatialDropout1D,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    SpatialDropout1D,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from preprocess import *

def build_cnn_base_model(
    X_train,
    y_train,
    X_val,
    y_val,
    embedding_matrix,
    maxlen,
    vocab_size,
    embedding_dim,
    learning_rate,
    epochs=3
):
    inp = Input(shape=(maxlen,))
    embedding = Embedding(
        vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False
    )(inp) # need to change
    
    # layers
    conv0 = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(
        embedding
    )
    pool0 = MaxPooling1D(pool_size=2)(conv0)

    conv1 = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(pool0)
    pool1 = MaxPooling1D(pool_size=2, padding="same")(conv1)

    conv2 = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(pool1)
    pool2 = MaxPooling1D(pool_size=2, padding='same')(conv2)

    conv3 = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(pool2)
    pool3 = MaxPooling1D(pool_size=2, padding='same')(conv3)

    flatten = Flatten()(pool3)
    dense0 = Dense(128, activation="relu")(flatten)
    dropout = Dropout(0.2)(dense0)
    dense1 = Dense(1, activation="sigmoid")(dropout)

    model = Model(inp, dense1)
    print(model.summary())

    # train model
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(lr=learning_rate),
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=256
    )

    return model, history
