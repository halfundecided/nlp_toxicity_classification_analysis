import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def preprocessing():
    df = pd.read_csv("../data/train.csv")

    comments = df["comment_text"].values
    targets = df["target"].values

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

    # one-hot encode for category labels --> can be removed later nate
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test_categorized = to_categorical(y_test)

    print("Shape of X_train", str(X_train.shape))
    print("Shape of y_train", str(y_train.shape))
    print("Shape of X_val", str(X_val.shape))
    print("Shape of y_val", str(y_val.shape))
    print("Shape of X_test", str(X_test.shape))
    print("Shape of y_test", str(y_test.shape))
    print("Shape of y_test_categorized", str(y_test_categorized.shape))

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_categorized


if __name__ == "__main__":
    preprocessing()