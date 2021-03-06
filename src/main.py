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
from bidirectional_lstm_att_fc import *

from metrics import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    EMBEDDING_DIM = 300  # need to add params to cnn and lstm
    EPOCHS = 6
    
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

    epoch_list = list(range(EPOCHS))
    fig, ((acc_ax, vacc_ax), (loss_ax, vloss_ax)) = plt.subplots(2,2)
    fig.suptitle("Training Analysis")
    fig.tight_layout(pad=5.0)
    ##### accuracy containers #####
    cnn1_acc = []
    cnn2_acc = []

    lstm1_acc = []
    lstm2_acc = []
    lstm3_acc = []

    bd_lstm1_acc = []
    bd_lstm2_acc = []
    bd_lstm3_acc = []

    ##### validation accuracy containers #####
    cnn1_acc_val = []
    cnn2_acc_val = []

    lstm1_acc_val = []
    lstm2_acc_val = []
    lstm3_acc_val = []

    bd_lstm1_acc_val = []
    bd_lstm2_acc_val = []
    bd_lstm3_acc_val = []

    ##### loss containers #####
    cnn1_loss = []
    cnn2_loss = []

    lstm1_loss = []
    lstm2_loss = []
    lstm3_loss = []

    bd_lstm1_loss = []
    bd_lstm2_loss = []
    bd_lstm3_loss = []

    ##### validation loss containers #####
    cnn1_loss_val = []
    cnn2_loss_val = []

    lstm1_loss_val = []
    lstm2_loss_val = []
    lstm3_loss_val = []

    bd_lstm1_loss_val = []
    bd_lstm2_loss_val = []
    bd_lstm3_loss_val = []

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
    cnn1_acc = cnn_base_history.history["accuracy"]
    cnn1_acc_val = cnn_base_history.history["val_accuracy"]
    cnn1_loss = cnn_base_history.history["loss"]
    cnn1_loss_val = cnn_base_history.history["val_loss"]

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
    cnn2_acc = cnn_sdbn_history.history["accuracy"]
    cnn2_acc_val = cnn_sdbn_history.history["val_accuracy"]
    cnn2_loss = cnn_sdbn_history.history["loss"]
    cnn2_loss_val = cnn_sdbn_history.history["val_loss"]

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

    lstm1_acc = lstm_base_history.history["accuracy"]
    lstm1_acc_val = lstm_base_history.history["val_accuracy"]
    lstm1_loss = lstm_base_history.history["loss"]
    lstm1_loss_val = lstm_base_history.history["val_loss"]

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

    lstm2_acc = lstm_2_history.history["accuracy"]
    lstm2_acc_val = lstm_2_history.history["val_accuracy"]
    lstm2_loss = lstm_2_history.history["loss"]
    lstm2_loss_val = lstm_2_history.history["val_loss"]

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

    lstm3_acc = lstm_att_history.history["accuracy"]
    lstm3_acc_val = lstm_att_history.history["val_accuracy"]
    lstm3_loss = lstm_att_history.history["loss"]
    lstm3_loss_val = lstm_att_history.history["val_loss"]

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

    bd_lstm1_acc = bd_lstm_history.history["accuracy"]
    bd_lstm1_acc_val = bd_lstm_history.history["val_accuracy"]
    bd_lstm1_loss = bd_lstm_history.history["loss"]
    bd_lstm1_loss_val = bd_lstm_history.history["val_loss"]

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

    bd_lstm2_acc = bd_lstm_att_history.history["accuracy"]
    bd_lstm2_acc_val = bd_lstm_att_history.history["val_accuracy"]
    bd_lstm2_loss = bd_lstm_att_history.history["loss"]
    bd_lstm2_loss_val = bd_lstm_att_history.history["val_loss"]

    ### Test Bidirectional LSTM ###
    bd_lstm_att_df = test_df.copy()
    test_pred = bd_lstm_att_model.predict(X_test)
    bd_lstm_att_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(bd_lstm_att_df, identity_cols, results_col, target_col)
    bd_lstm_att_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(bd_lstm_att_df, results_col, target_col)))
    bd_lstm_att_loss,_ = bd_lstm_att_model.evaluate(X_test, y_test)
    print("Bidirectional LSTM Attention Bias Score: {:.4f} --- Bidirectional LSTM Attention Loss: {:.4f}".format(bd_lstm_att_score, bd_lstm_att_loss))



    ##### Build BD-LSTM_Attention Model #####
    bd_lstm_att_fc_model, bd_lstm_att_fc_history = build_bidirectional_att_fc_model(
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

    bd_lstm3_acc = bd_lstm_att_fc_history.history["accuracy"]
    bd_lstm3_acc_val = bd_lstm_att_fc_history.history["val_accuracy"]
    bd_lstm3_loss = bd_lstm_att_fc_history.history["loss"]
    bd_lstm3_loss_val = bd_lstm_att_fc_history.history["val_loss"]

    ### Test Bidirectional LSTM ###
    bd_lstm_att_fc_df = test_df.copy()
    test_pred = bd_lstm_att_fc_model.predict(X_test)
    bd_lstm_att_fc_df[results_col] = test_pred

    bias_metrics_df = compute_bias_metrics_for_model(bd_lstm_att_fc_df, identity_cols, results_col, target_col)
    bd_lstm_att_fc_score = (get_final_metric(bias_metrics_df, calculate_overall_auc(bd_lstm_att_fc_df, results_col, target_col)))
    bd_lstm_att_fc_loss,_ = bd_lstm_att_fc_model.evaluate(X_test, y_test)
    print("Bidirectional LSTM Attention Bias Score: {:.4f} --- Bidirectional LSTM Attention Loss: {:.4f}".format(bd_lstm_att_fc_score, bd_lstm_att_fc_loss))


    ##### Plot Data #####
    # accuracy
    acc_ax.plot(epoch_list, cnn1_acc, label="CNN")
    acc_ax.plot(epoch_list, cnn2_acc, label="CNN+(SD,BN)")
    acc_ax.plot(epoch_list, lstm1_acc, label="LSTM")
    acc_ax.plot(epoch_list, lstm2_acc, label="LSTM+(x2)")
    acc_ax.plot(epoch_list, lstm3_acc, label="LSTM+(x2,ATT)")
    acc_ax.plot(epoch_list, bd_lstm1_acc, label="B-LSTM+(x2)")
    acc_ax.plot(epoch_list, bd_lstm2_acc, label="B-LSTM+(x2,ATT)")
    acc_ax.plot(epoch_list, bd_lstm3_acc, label="B-LSTM+(x2,ATT,FC)")

    acc_ax.set_title("Accuracy")
    acc_ax.set_xlabel("Iterations")
    acc_ax.set_ylabel("Accuracy")

    # loss
    loss_ax.plot(epoch_list, cnn1_loss)
    loss_ax.plot(epoch_list, cnn2_loss)
    loss_ax.plot(epoch_list, lstm1_loss)
    loss_ax.plot(epoch_list, lstm2_loss)
    loss_ax.plot(epoch_list, lstm3_loss)
    loss_ax.plot(epoch_list, bd_lstm1_loss)
    loss_ax.plot(epoch_list, bd_lstm2_loss)
    loss_ax.plot(epoch_list, bd_lstm3_loss)

    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Iterations")
    loss_ax.set_ylabel("Loss")
    
    # val accuracy
    vacc_ax.plot(epoch_list, cnn1_acc_val)
    vacc_ax.plot(epoch_list, cnn2_acc_val)
    vacc_ax.plot(epoch_list, lstm1_acc_val)
    vacc_ax.plot(epoch_list, lstm2_acc_val)
    vacc_ax.plot(epoch_list, lstm3_acc_val)
    vacc_ax.plot(epoch_list, bd_lstm1_acc_val)
    vacc_ax.plot(epoch_list, bd_lstm2_acc_val)
    vacc_ax.plot(epoch_list, bd_lstm3_acc_val)

    vacc_ax.set_title("Validation Accuracy")
    vacc_ax.set_xlabel("Iterations")
    vacc_ax.set_ylabel("Accuracy")

    # val loss
    vloss_ax.plot(epoch_list, cnn1_loss_val)
    vloss_ax.plot(epoch_list, cnn2_loss_val)
    vloss_ax.plot(epoch_list, lstm1_loss_val)
    vloss_ax.plot(epoch_list, lstm2_loss_val)
    vloss_ax.plot(epoch_list, lstm3_loss_val)
    vloss_ax.plot(epoch_list, bd_lstm1_loss_val)
    vloss_ax.plot(epoch_list, bd_lstm2_loss_val)
    vloss_ax.plot(epoch_list, bd_lstm3_loss_val)

    vloss_ax.set_title("Validation Loss")
    vloss_ax.set_xlabel("Iterations")
    vloss_ax.set_ylabel("Loss")

    fig.savefig("train_analysis.png",bbox_inches="tight")
