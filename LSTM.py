from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K

from DataProcess import DataProcess
from Sender import Sender

import pandas as pd
import datetime
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LSTMFunc(object):
    def __init__(self, train_x, train_y, test_x, test_y, scaler):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.scaler = scaler

    def func(self):
        # design network
        model = Sequential()
        model.add(LSTM(768, input_shape=(self.train_x.shape[1], self.train_x.shape[2]), return_sequences=True))
        model.add(LSTM(384, return_sequences=True))
        model.add(LSTM(96, return_sequences=False))
        model.add(Dense(96))
        model.compile(loss='mse', optimizer='adam')
        model.fit(self.train_x, self.train_y, epochs=100, batch_size=365, validation_data=(self.test_x,self.test_y),
                  verbose=0,shuffle=False)

        get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[3].output])
        layer_output = get_3rd_layer_output([self.train_x])[0]
        out = self.scaler.inverse_transform(layer_output)[-1]

        return out

if __name__ == "__main__":
    # data path
    baseDir =  "C:/Users/bios/Desktop/algorithm-control-center/data/elecData/1923270.csv" #"data/elecData/1923270.csv"
    startTime = datetime.datetime.now()
    df1 = pd.read_csv(baseDir, header=0, usecols=[i for i in range(0,100)])
    df1.drop(['ycid','a','b'],axis=1,inplace=True)

    loss = sys.argv[1]
    optimizer = sys.argv[2]
    userId = sys.argv[3]
    trainStart = sys.argv[4].replace("-0", "/").replace("-", "/")
    trainEnd = sys.argv[5].replace("-0", "/").replace("-", "/")
    print(sys.argv)
    #print(df1[(df1.time == sys.argv[4])].index)
    #print(df1[(df1.time == trainStart)].index)
    idxStart = df1[(df1.time == trainStart)].index.tolist()[0]
    idxEnd = df1[(df1.time == trainEnd)].index.tolist()[0]
    #df1.drop(['time'], axis=1, inplace=True)
    #print(df1.columns)
    preData = DataProcess(df1.ix[idxStart:idxEnd+1, [i for i in range(1, 97)]])
    train_x, train_y, test_x, test_y, scaler = preData.generate_data()
    out = LSTMFunc(train_x, train_y, test_x, test_y, scaler).func()
    nout = []
    for e in out:
        nout.append(float('%.2f' % e))
    # out = '[1.65,1.59,1.59,1.43,1.56,1.49,1.45,1.36,1.44,1.49,1.61,1.56,1.82,1.79,1.89,1.91,1.92,1.84,1.88,1.77,1.82,1.65,1.68,1.45,1.50,1.55,1.47,1.53,1.54,1.57,1.60,1.68,1.69,1.70,1.79,1.75,1.76,1.76,1.85,1.73,1.98,2.09,2.23,2.11,2.08,1.78,1.74,1.77,1.81,1.71,1.79,1.66,1.75,1.69,1.54,1.62,1.69,1.62,1.69,1.62,1.61,1.57,1.69,1.68,1.95,2.07,2.18,2.02,1.93,1.70,1.79,1.77,1.83,1.79,1.96,1.87,1.88,1.93,1.91,1.85,1.94,1.85,2.00,1.93,1.99,1.97,1.93,1.91,1.93,1.84,1.87,1.78,1.81,1.71,1.74,1.66]'

    endTime = datetime.datetime.now()
    interval = str('%.2f' % (endTime-startTime).total_seconds())
    params = loss+" "+optimizer+" "+sys.argv[3]+" "+sys.argv[4]
    # params = "mse adam 1923270 2014-01-01 2016-04-30"

    twoWeeksBefore = ""
    for e in df1[idxEnd-13:idxEnd+1].values:
        twoWeeksBefore += str(e.tolist())

    #twoDaysBefore = df1[idxEnd-2:idxEnd+1, :]
    startTime = startTime.strftime("%Y-%m-%d %H:%M:%S")
    endTime = endTime.strftime("%Y-%m-%d %H:%M:%S")
    msg = "LSTM|"+userId+"|"+sys.argv[5]+"|"+str(nout)+"|"+twoWeeksBefore+"|"+startTime+"|"+endTime+"|"+interval+"|"+params+"|-1"
    # msg = "1923270 at 2016-04-30|[1.64721417,1.58620143,1.58976829,1.42793834,1.56486976,1.48822665,1.45411265,1.36233544,1.43914711,1.49116325,1.60853636,1.56123328,1.81846237,1.78751326,1.88805366,1.90746474,1.9210906,1.83574867,1.87868893,1.76570761,1.82136774,1.65228164,1.67854679,1.45034373,1.50406373,1.54984689,1.46876061,1.53323817,1.5403111,1.57380235,1.60259974,1.68235123,1.68979001,1.69517267,1.78855991,1.74946141,1.75662351,1.76206374,1.8470031, 1.73198855,1.97918034,2.09072232,2.23420382,2.11208582,2.07506132,1.78128028,1.74324095,1.76831245,1.80583572,1.70506775,1.79009473,1.65794909,1.74614203,1.68759191,1.53605413,1.61827767,1.68517339,1.61839998,1.68937159,1.61886263,1.60974717,1.57182634,1.68984842,1.67505097,1.9481138, 2.07150817,2.17925763,2.02420521,1.93154812,1.6997354, 1.79060924,1.76898694,1.83197486,1.79223168,1.96496749,1.86803293,1.87520492,1.93327105,1.91270232,1.85107911,1.94007993,1.84738219,1.99670184,1.93224537,1.98743331,1.97292304,1.92522919,1.91341054,1.93055236,1.84190869,1.86796486,1.78458047,1.80774117,1.70798671,1.74037731,1.65715051]|"+startTime+"|"+endTime+"|"+interval+"|LSTM|"+params+"|-1"
    print("lstm->", msg)
    Sender().sendMessage(msg)