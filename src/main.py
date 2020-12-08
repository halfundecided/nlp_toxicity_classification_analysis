import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress info output from tf

# configure GPU settings
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# append path to models and layers
sys.path.append("./models/")
sys.path.append("./layers/")

from preprocess import *

from cnn_base import *
from cnn_sdbn import *

from lstm_base import *
from lstm_2 import *
from lstm_att import *

from bidirectional_lstm import *
from bidirectional_lstm_att import *

from metrics import *

if __name__ == "__main__":
    EMBEDDING_DIM = 300  # need to add params to cnn and lstm
    EPOCHS = 3
    
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

    ##### Define Testing Variables #####
    identity_cols = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    results_col = 'predicted_toxicity'
    target_col = 'toxicity'

    ##### Build CNN_Base Model #####
    cnn_base_model, cnn_base_history = build_cnn_base_model(
        X_train,
        y_train,
        X_val,
        y_val,
        embedding_matrix,
        maxlen,
        vocab_size,
        EMBEDDING_DIM,
        learning_rate=1e-3,
        epochs=EPOCHS
    )

    ### Test CNN_Base ###
    cnn_base_df = test_df.copy()
    test_pred = cnn_base_model.predict(X_test)
    cnn_base_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(cnn_base_df, identity_cols, results_col, target_col)
    cnn_base_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(cnn_base_df, results_col, target_col)))
    cnn_base_loss,_ = cnn_base_model.evaluate(X_test, y_test)
    print("CNN_Base Bias Score: {:.4f} --- CNN_Base Loss: {:.4f}".format(cnn_base_score, cnn_base_loss))



    ##### Build CNN_SDBN Model #####
    cnn_sdbn_model, cnn_sdbn_history = build_cnn_sdbn_model(
        X_train,
        y_train,
        X_val,
        y_val,
        embedding_matrix,
        maxlen,
        vocab_size,
        EMBEDDING_DIM,
        learning_rate=1e-3,
        epochs=EPOCHS
    )

    ### Test CNN_Base ###
    cnn_sdbn_df = test_df.copy()
    test_pred = cnn_sdbn_model.predict(X_test)
    cnn_sdbn_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(cnn_sdbn_df, identity_cols, results_col, target_col)
    cnn_sdbn_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(cnn_sdbn_df, results_col, target_col)))
    cnn_sdbn_loss,_ = cnn_sdbn_model.evaluate(X_test, y_test)
    print("CNN_SBDN Bias Score: {:.4f} --- CNN_SBDN Loss: {:.4f}".format(cnn_sdbn_score, cnn_sdbn_loss))



    ##### Build LSTM_Base Model #####
    lstm_base_model, lstm_base_history = build_lstm_base_model(
    X_train,
    y_train,
    X_val,
    y_val,
    embedding_matrix,
    vocab_size,
    EMBEDDING_DIM,
    learning_rate=1e-3,
    epochs=EPOCHS)

    ### Test LSTM_Base ###
    lstm_base_df = test_df.copy()
    test_pred = lstm_base_model.predict(X_test)
    lstm_base_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(lstm_base_df, identity_cols, results_col, target_col)
    lstm_base_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(lstm_base_df, results_col, target_col)))
    lstm_base_loss,_ = lstm_base_model.evaluate(X_test, y_test)
    print("LSTM_Base Bias Score: {:.4f} --- LSTM_Base Loss: {:.4f}".format(lstm_base_score, lstm_base_loss))



    ##### Build LSTM_2 Model #####
    lstm_2_model, lstm_2_history = build_lstm_2_model(
    X_train,
    y_train,
    X_val,
    y_val,
    embedding_matrix,
    vocab_size,
    EMBEDDING_DIM,
    learning_rate=1e-3,
    epochs=EPOCHS)

    ### Test LSTM_2 ###
    lstm_2_df = test_df.copy()
    test_pred = lstm_2_model.predict(X_test)
    lstm_2_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(lstm_2_df, identity_cols, results_col, target_col)
    lstm_2_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(lstm_2_df, results_col, target_col)))
    lstm_2_loss,_ = lstm_2_model.evaluate(X_test, y_test)
    print("LSTM_2 Bias Score: {:.4f} --- LSTM_2 Loss: {:.4f}".format(lstm_2_score, lstm_2_loss))


    ##### Build LSTM_Attention Model #####
    lstm_att_model, lstm_att_history = build_lstm_att_model(
        X_train,
        y_train,
        X_val,
        y_val,
        embedding_matrix,
        maxlen,
        vocab_size,
        EMBEDDING_DIM,
        learning_rate=1e-3,
        epochs=EPOCHS
    )

    ### Test LSTM_Attention ###
    lstm_att_df = test_df.copy()
    test_pred = lstm_att_model.predict(X_test)
    lstm_att_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(lstm_att_df, identity_cols, results_col, target_col)
    lstm_att_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(lstm_att_df, results_col, target_col)))
    lstm_att_loss,_ = lstm_att_model.evaluate(X_test, y_test)
    print("LSTM_Attention Bias Score: {:.4f} --- LSTM_Attention Loss: {:.4f}".format(lstm_att_score, lstm_att_loss))


    
    ##### Build BD-LSTM Model #####
    bd_lstm_model, bd_lstm_history = build_bidirectional_model(
    X_train,
    y_train,
    X_val,
    y_val,
    embedding_matrix,
    maxlen,
    vocab_size,
    EMBEDDING_DIM,
    learning_rate=1e-3,
    epochs=EPOCHS)

    ### Test Bidirectional LSTM ###
    bd_lstm_df = test_df.copy()
    test_pred = bd_lstm_model.predict(X_test)
    bd_lstm_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(bd_lstm_df, identity_cols, results_col, target_col)
    bd_lstm_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(bd_lstm_df, results_col, target_col)))
    bd_lstm_loss,_ = bd_lstm_model.evaluate(X_test, y_test)
    print("Bidirectional LSTM Bias Score: {:.4f} --- Bidirectional LSTM Loss: {:.4f}".format(bd_lstm_score, bd_lstm_loss))



    
    ##### Build BD-LSTM_Attention Model #####
    bd_lstm_att_model, bd_lstm_att_history = build_bidirectional_att_model(
    X_train,
    y_train,
    X_val,
    y_val,
    embedding_matrix,
    maxlen,
    vocab_size,
    EMBEDDING_DIM,
    learning_rate=1e-3,
    epochs=EPOCHS)

    ### Test Bidirectional LSTM ###
    bd_lstm_att_df = test_df.copy()
    test_pred = bd_lstm_att_model.predict(X_test)
    bd_lstm_att_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(bd_lstm_att_df, identity_cols, results_col, target_col)
    bd_lstm_att_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(bd_lstm_att_df, results_col, target_col)))
    bd_lstm_att_loss,_ = bd_lstm_att_model.evaluate(X_test, y_test)
    print("Bidirectional LSTM Attention Bias Score: {:.4f} --- Bidirectional LSTM Attention Loss: {:.4f}".format(bd_lstm_att_score, bd_lstm_att_loss))
