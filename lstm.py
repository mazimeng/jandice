from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense
from multiprocessing.dummy import Pool

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from numpy import newaxis


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


class DataLoader(object):
    """A class for loading and transforming data for the lstm model"""

    def __init__(self):
        # dataframe = pd.read_csv(filename)
        # i_split = int(len(dataframe) * split)
        # self.data_train = dataframe.get(cols).values[:i_split]
        # self.data_test = dataframe.get(cols).values[i_split:]
        # self.len_train = len(self.data_train)
        # self.len_test = len(self.data_test)
        self.len_train_windows = None
        self.data_train = None
        self.data_test = None
        self.len_train = 0
        self.len_test = 0

    def load_training_data(self, filename, cols):
        dataframe = pd.read_csv(filename)
        self.data_train = dataframe.get(cols).values
        self.len_train = len(self.data_train)

    def load_testing_data(self, filename, cols):
        dataframe = pd.read_csv(filename)
        self.data_test = dataframe.get(cols).values
        self.len_test = len(self.data_test)

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i + seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = []
                for p in window[:, col_i]:
                    v = float(window[0, col_i])
                    if v > 0:
                        v = (float(p) / v) - 1
                    normalised_col.append(v)
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        # plt.legend()
    plt.show()


def main():
    model = Sequential()

    sequence_length = 50

    data_loader = DataLoader()
    data_loader.load_training_data("data/bars_training.csv", ["close", "volume"])
    data_loader.load_testing_data("data/bars_testing.csv", ["close", "volume"])

    saved_model_file = "files/20181018-192815-e500.h5"
    # saved_model_file = None
    do_training = False
    do_predict = True

    if saved_model_file:
        model = load_model(saved_model_file)
    else:
        model.add(LSTM(100, input_shape=(sequence_length - 1, 2), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100, input_shape=(None, None), return_sequences=True))
        model.add(LSTM(100, input_shape=(None, None), return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mse", optimizer="adam")

    if do_training:
        epochs = 500
        training_batch_size = 2000

        x, y = data_loader.get_train_data(sequence_length, True)

        if not saved_model_file:
            saved_model_file = 'files/%s-e%s.h5' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), str(epochs))

        callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                     ModelCheckpoint(filepath=saved_model_file, monitor='val_loss', save_best_only=True)]

        model.fit(x,
                  y,
                  epochs=epochs,
                  batch_size=training_batch_size,
                  callbacks=callbacks)

        model.save(saved_model_file)

    if do_predict:
        x_test, y_test = data_loader.get_test_data(
            seq_len=sequence_length,
            normalise=True)

        # predicted = model.predict(x_test)
        # predicted = np.reshape(predicted, (predicted.size, 1))
        # plot_results(predicted, y_test)

        predictions = predict_sequences_multiple(model, x_test, sequence_length, 10)
        plot_results_multiple(predictions, y_test, 10)


if __name__ == '__main__':
    main()
