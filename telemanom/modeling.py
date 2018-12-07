from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
import numpy as np
import os
from telemanom._globals import Config

#config
config = Config("config.yaml")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def get_model(anom, X_train, y_train, logger, train=False):


    if not train and os.path.exists(os.path.join("data", "models", anom["chan_id"] + ".h5")):
        logger.info("Загрузка модели")
        return load_model(os.path.join("data", config.use_id, "models", anom["chan_id"] + ".h5"))

    elif (not train and not os.path.exists(os.path.join("data", config.use_id, "models", anom["chan_id"] + ".h5"))) or train:
        
        if not train:
            logger.info("Обучение новой модели.")

        cbs = [History(), EarlyStopping(monitor='val_loss', patience=config.patience, 
            min_delta=config.min_delta, verbose=0)]
        
        model = Sequential()

        model.add(LSTM(
            config.layers[0],
            input_shape=(None, X_train.shape[2]),
            return_sequences=True))
        model.add(Dropout(config.dropout))

        model.add(LSTM(
            config.layers[1],
            return_sequences=False))
        model.add(Dropout(config.dropout))

        model.add(Dense(
            config.n_predictions))
        model.add(Activation("linear"))

        model.compile(loss=config.loss_metric, optimizer=config.optimizer) 

        model.fit(X_train, y_train, batch_size=config.lstm_batch_size, epochs=config.epochs, 
            validation_split=config.validation_split, callbacks=cbs, verbose=True)
        model.save(os.path.join("data", anom['run_id'], "models", anom["chan_id"] + ".h5"))

        return model




def predict_in_batches(y_test, X_test, model, anom):

    y_hat = np.array([])

    num_batches = int((y_test.shape[0] - config.l_s) / config.batch_size)
    if num_batches < 0:
        raise ValueError("l_s (%s) слишком большое знвчение для потока длиной %s." %(config.l_s, y_test.shape[0]))

    # simulate data arriving in batches
    for i in range(1, num_batches+2):
        prior_idx = (i-1) * config.batch_size
        idx = i * config.batch_size
        if i == num_batches+1:
            idx = y_test.shape[0]
        
        X_test_period = X_test[prior_idx:idx]

        y_hat_period = model.predict(X_test_period)


        final_y_hat = []
        for t in range(len(y_hat_period)+config.n_predictions):
            y_hat_t = []
            for j in range(config.n_predictions):
                if t - j >= 0 and t-j < len(y_hat_period):
                    y_hat_t.append(y_hat_period[t-j][j])
            if t < len(y_hat_period):
                if y_hat_t.count(0) == len(y_hat_t):
                    final_y_hat.append(0)
                else:
                    final_y_hat.append(y_hat_t[0]) # first prediction


        y_hat_period = np.array(final_y_hat).reshape(len(final_y_hat),1)
        y_hat = np.append(y_hat, y_hat_period)

    y_hat = np.reshape(y_hat, (y_hat.size,))

    np.save(os.path.join("data", anom['run_id'], "y_hat", anom["chan_id"] + ".npy"), np.array(y_hat))

    return y_hat