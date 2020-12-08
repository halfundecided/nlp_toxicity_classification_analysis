from tensorflow.keras.layers import (
    Input,
    Embedding,
    Concatenate,
    Add,
    SpatialDropout1D,
    Bidirectional,
    LSTM,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    Dropout,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from attention import *

def build_bidirectional_att_fc_model(
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
    input = Input(shape=(maxlen,))
    embedding = Embedding(
        vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False
    )(input)
    spatial_dropout = SpatialDropout1D(0.2)(embedding)
    # CuDNNLSTM: https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/layers/CuDNNLSTM
    bilstm0 = Bidirectional(LSTM(128, return_sequences=True))(spatial_dropout)
    bilstm1 = Bidirectional(LSTM(128, return_sequences=True))(bilstm0)
    att = Attention(maxlen) (bilstm1)

    poolings = Concatenate() (
        [att, GlobalMaxPooling1D()(bilstm1), GlobalAveragePooling1D()(bilstm1)]
    )

    hidden0 = Add()([poolings, Dense(768, activation="relu")(poolings)])
    hidden1 = Add()([hidden0, Dense(768, activation="relu")(hidden0)])

    fc1 = Dense(1024, activation="relu") (hidden1)
    dropout1 = Dropout(0.4) (fc1)
    fc2 = Dense(512, activation="relu") (dropout1)

    dense = Dense(1, activation="sigmoid")(fc2)

    model = Model(input, dense)
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
