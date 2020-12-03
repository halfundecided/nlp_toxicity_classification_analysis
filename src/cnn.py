import numpy as np
import sys
import os

# suppress info output from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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
    input = Input(
        shape(
            maxlen,
        )
    )
    embedding = Embedding(vocan_size, 100)  # need to change

    # dropout = SpatialDropout1D(rate=0.0)(embedding)
    # attention

    # layers
    conv0 = Conv1D(filters=64, kernel_size=2, padding="same", activation="relu")(
        dropout
    )
    pool0 = MaxPooling1D(pool_size=5)(conv0)

    conv1 = Conv1D(filters=128, kernel_size=2, padding="same", activation="relu")(pool0)
    pool1 = MaxPooling1D(pool_size=5)(conv1)

    conv2 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(pool1)
    pool2 = MaxPooling1D(pool_size=5)(conv2)

    conv3 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(pool2)
    pool3 = MaxPooling1D(pool_size=5)(conv3)

    flatten = Flatter()(pool3)
    dense0 = Dense(64, activation="relu")(flatten)
    dropout = Dropout(0.2)(dense0)
    dense1 = Dense(3, activation="softmax")(dropout)

    model = Model(input, dense1)
    print(model.summary())

    # train model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(lr=learning_rate),
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=128
    )

    return model


if __name__ == "__main__":
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        y_test_categorized,
        maxlen,
        vocab_size,
    ) = preprocessing()
    model = build_mode(
        X_train, y_train, X_val, y_val, maxlen, vocab_size, learning_rate=1e-3
    )
    loss_and_acc = model.evaluate(X_test, y_test_categorized)
    print(loss_and_acc[1])
