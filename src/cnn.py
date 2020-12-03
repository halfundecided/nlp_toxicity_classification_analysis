import numpy as np
import sys
import os

# suppress info output from tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from keras.layers import (
    Input,
    Embedding,
    SpatialDropout1D,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
)
from keras.models import Model
from keras import optimizers

from preprocess import *


def build_model(X_train, y_train, X_val, y_val, maxlen, vocab_size, learning_rate):
    input = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(
        input
    )  # need to change

    # layers
    conv0 = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(
        embedding
    )
    pool0 = MaxPooling1D(pool_size=5)(conv0)

    conv1 = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(pool0)
    pool1 = MaxPooling1D(pool_size=5, padding="same")(conv1)

    # conv2 = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(pool1)
    # pool2 = MaxPooling1D(pool_size=5, padding='same')(conv2)

    # conv3 = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(pool2)
    # pool3 = MaxPooling1D(pool_size=5, padding='same')(conv3)

    flatten = Flatten()(pool1)
    dense0 = Dense(64, activation="relu")(flatten)
    dropout = Dropout(0.2)(dense0)
    dense1 = Dense(1, activation="sigmoid")(dropout)

    model = Model(input, dense1)
    print(model.summary())

    # train model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(lr=learning_rate),
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train, epochs=3, validation_data=(X_val, y_val), batch_size=128
    )

    return model
