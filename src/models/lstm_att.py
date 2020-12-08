from tensorflow.keras.layers import (
    Input,
    Embedding,
    SpatialDropout1D,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Concatenate,
    LSTM,
    Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

import sys

from attention import *

def build_lstm_att_model(
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
    embedd = Embedding(
            vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False
        ) (inp)
    sdo = SpatialDropout1D(0.2) (embedd)
    lstm0 = LSTM(128, dropout=0.2, return_sequences=True) (sdo)
    lstm1 = LSTM(128, dropout=0.2, return_sequences=True) (lstm0)

    att = Attention(maxlen) (lstm1)
    avg_pool = GlobalAveragePooling1D() (lstm1)
    max_pool = GlobalMaxPooling1D() (lstm1)

    cat = Concatenate() ([att, avg_pool, max_pool])

    dense = Dense(1, activation="sigmoid") (cat)

    model = Model(inp, dense)
    model.summary()
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
