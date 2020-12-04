import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress info output from tf

# configure GPU settings
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from preprocess import *
from cnn import *
from lstm import *
from metrics import *

if __name__ == "__main__":
    EMBEDDING_DIM = 300  # need to add params to cnn and lstm

    ##### Run Preprocessing #####
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        test_df,
        maxlen,
        vocab_size,
        word_index,
    ) = preprocessing()
    print("done processing!")

    ##### Build Embedding Matrix #####
    embedding_matrix = glove_embedding(
        "../data/glove.42B.300d.txt", vocab_size, word_index
    )
    print("done embedding!")

    ##### Build CNN Model #####
    cnn_model, cnn_history = build_cnn_model(
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

    ##### Build RNN Model #####
    lstm_model, lstm_history = build_lstm_model(
    X_train,
    y_train,
    X_val,
    y_val,
    embedding_matrix,
    vocab_size,
    EMBEDDING_DIM,
    learning_rate=1e-3)

    ##### Test against custom bias metric #####
    identity_cols = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    results_col = 'predicted_toxicity'
    target_col = 'toxicity'

    cnn_df = test_df.copy()
    test_pred = cnn_model.predict(X_test)
    cnn_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(cnn_df, identity_cols, results_col, target_col)
    cnn_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(cnn_df, results_col, target_col)))
    print("CNN Bias Score:",cnn_score)


    lstm_df = test_df.copy()
    test_pred = lstm_model.predict(X_test)
    lstm_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(lstm_df, identity_cols, results_col, target_col)
    lstm_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(lstm_df, results_col, target_col)))
    print("LSTM Bias Score:",lstm_score)
