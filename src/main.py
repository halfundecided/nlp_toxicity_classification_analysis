from preprocess import *
from cnn import *

if __name__ == "__main__":
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
    embedding_matrix = glove_embedding("", vocab_size, word_index)

    model = build_model(
        X_train, y_train, X_val, y_val, maxlen, vocab_size, learning_rate=1e-3
    )
    loss_and_acc = model.evaluate(X_test, y_test)
    print(loss_and_acc[1])
