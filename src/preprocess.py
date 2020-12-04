import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import sys

def preprocessing():
    # read in training and testing data
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test_public_expanded.csv")

    # replace all NA/NaN values with empty-string
    train_df['comment_text'] = train_df['comment_text'].fillna('')
    test_df['comment_text'] = test_df['comment_text'].fillna('')

    # grab comments from data frame
    train_comments = list(train_df['comment_text'].values)
    test_comments = list(test_df['comment_text'].values)
    all_comments = train_comments + test_comments

    # convert target labels to True(toxic)/False(non-toxic) (ps. they use 'toxicity' as 'target' column in test data)
    identity_cols = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

    y_train = np.where(train_df["target"] >= 0.5, True, False) * 1
    y_test = np.where(test_df["toxicity"] >= 0.5, True, False) * 1
    for c in identity_cols+["toxicity"]:
        test_df[c] = np.where(test_df[c] >= 0.5, True, False)


    # split training into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_comments, y_train, test_size=0.15, shuffle=True
    )

    # create tokenizer for data
    tokenizer = Tokenizer(num_words=250000)
    tokenizer.fit_on_texts(all_comments)

    # convert text to tokens
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_test = tokenizer.texts_to_sequences(test_comments)

    # grab vocab size and max sentence length
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 100

    # pad/truncate all the data to maxlen
    X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
    X_val = pad_sequences(X_val, padding="post", maxlen=maxlen)
    X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)

    print("Shape of X_train", X_train.shape)
    print("Shape of y_train", y_train.shape)
    print("Shape of X_val", X_val.shape)
    print("Shape of y_val", y_val.shape)
    print("Shape of X_test", X_test.shape)
    print("Shape of y_test", y_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        test_df,
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
