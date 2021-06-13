from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class DataProcess(object):
    def __init__(self, data):
        self.data = data

    # change time series to supervised learning problem
    def series_to_supervised(self, data, n_in = 1, n_out=1, dropnan=True):
        n_vars=1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()

        # input sequence(t-n, , , t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names = names+[('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        # predict sequence(t, t+1, , t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names = names+[('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names = names+[('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        agg = pd.concat(cols, axis = 1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace = True)
        return agg

    def generate_data(self):
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(self.data)
        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, 1, 1)

        # data of 2014, 2015 used for fit the model, 2016 to evaluate
        # split to train set and test set
        values = reframed.values
        n_train_hours = 851*15
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]

        # split to input and output variable
        train_x, train_y = train[:, :-96], train[:, 96:]
        test_x, test_y = test[:, :-96], test[:, 96:]

        # reshape input to 3D[samples, timesteps, features]
        train_x=train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
        test_x=test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

        return train_x, train_y, test_x, test_y, scaler
