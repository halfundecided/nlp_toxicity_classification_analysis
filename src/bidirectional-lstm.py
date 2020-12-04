from tensorflow.keras.layers import (
    Embedding,
    SpatialDropout1D,
    Bidirectional,
    CuDNNLSTM,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
)
from tensorflow.kerals.models import Sequential
from tensorflow.keras import optimizers


def build_bidirectional_model(
    X_train,
    y_train,
    X_val,
    y_val,
    embedding_matrix,
    maxlen,
    vocab_size,
    embedding_dim,
    learning_rate,
):
    input = Input(shape=(maxlen,))
    embedding = Embedding(
        vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False
    )(input)
    spatial_dropout = SpatialDropout1D(0.2)(embedding)
    # CuDNNLSTM: https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/layers/CuDNNLSTM
    bilstm0 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(spatial_dropout)
    bilstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(bilstm0)

    poolings = concatenate(
        [GlobalMaxPooling1D()(bilstm1), GlobalAveragePooling1D()(bilstm1)]
    )

    hidden0 = add([poolings, Dense(512, activation="relu")(poolings)])
    hidden1 = add([hidden0, Dense(512, activation="relu")(hidden0)])
    dense = Dense(1, activation="sigmoid")(hidden1)

    model = Model(input, dense)
    print(model.summary())

    # train model
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer.Adam(lr=learning_rate),
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train, y_train, epochs=3, validation_data=(X_val, y_val), batch_size=128
    )

    return model, history
