import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress info output from tf
from preprocess import *
from cnn import *
from lstm import *

if __name__ == "__main__":
    EMBEDDING_DIM = 300  # need to add params to cnn and lstm
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        maxlen,
        vocab_size,
        word_index,
    ) = preprocessing()
    print("done processing!")
    embedding_matrix = glove_embedding(
        "../data/glove.42B.300d.txt", vocab_size, word_index
    )
    print("done embedding!")
    model = build_cnn_model(
        X_train,
        y_train,
        X_val,
        y_val,
        embedding_matrix,
        maxlen,
        vocab_size,
        EMBEDDING_DIM,
        learning_rate=1e-3,
    )
    loss_and_acc = model.evaluate(X_test, y_test)
    print(loss_and_acc[1])
