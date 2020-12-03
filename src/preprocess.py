import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def preprocessing():
    df = pd.read_csv("../data/train.csv")

    comments = df["comment_text"].values
    targets = np.where(df["target"] >= 0.5, True, False) * 1

    comments_train, comments_test, y_train, y_test = train_test_split(
        comments, targets, test_size=0.25, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        comments_train, y_train, test_size=0.05, shuffle=True
    )

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_test = tokenizer.texts_to_sequences(comments_test)

    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 100

    X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
    X_val = pad_sequences(X_val, padding="post", maxlen=maxlen)
    X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)

    print("Shape of X_train", str(X_train.shape))
    print("Shape of y_train", str(y_train.shape))
    print("Shape of X_val", str(X_val.shape))
    print("Shape of y_val", str(y_val.shape))
    print("Shape of X_test", str(X_test.shape))
    print("Shape of y_test", str(y_test.shape))

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        maxlen,
        vocab_size,
        tokenizer.word_index,
    )


def glove_embedding(file_path, vocab_size, word_index):
    embedding_index = {}
    f = open(file_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("The shape of embedding matrix: ", str(embedding_matrix.shape))

    return embedding_matrix
