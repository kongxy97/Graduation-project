import datetime
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from LassoPre import LassoPre
from LassoStatistics import LassoStatistics


class LASSOFunc(object):
    def __init__(self, base_dir, user_id, train_start_date, train_end_date):
        self.base_dir = base_dir
        self.user_id = user_id
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

    def func(self):
        pred = []
        real = []
        train_score = []
        test_score = []
        for i in range(1, 97):
            file_dir = self.base_dir+str(i)+".csv"
            #print(file_dir)
            df2 = pd.read_csv(file_dir, header=0, index_col=0)

            if np.isnan(df2.iloc[-1, -1]):  # the last row may be null
                column_x1 = df2.iloc[:-1, 0:-1]
                column_y1 = df2.iloc[:-1, -1]
            else:
                column_x1 = df2.iloc[:, 0:-1]
                column_y1 = df2.iloc[:, -1]

            x_train, x_test, y_train, y_test = train_test_split(column_x1, column_y1, test_size=0.2, random_state=42)
            lasso = Lasso(alpha=0.15)
            lasso.fit(x_train, y_train)
            # train_score = lasso.score(x_train, y_train)
            # test_score = lasso.score(x_test, y_test)
            train_score.append(float('%.2f' % lasso.score(x_train, y_train)))
            test_score.append(float('%.2f' % lasso.score(x_test, y_test)))
            x_pre = column_x1.iloc[-7:-1]
            y_rel = column_y1.iloc[-7:-1]
            y_pre = lasso.predict(x_pre)
            # print("mean test=",y_rel.mean(),", pred=",y_pre.mean())
            real.append(float('%.2f' % y_rel.mean()))
            pred.append(float('%.2f' % y_pre.mean()))
        return pred, real, train_score, test_score

if __name__ == "__main__":
    user_id = sys.argv[1]
    train_start_date = sys.argv[2].replace("-0", "/").replace("-", "/")
    train_end_date = sys.argv[3].replace("-0", "/").replace("-", "/")
    start_time = datetime.datetime.now()
    base_dir = "data/elecData/"+user_id
    # timestep's file existence, if not then produce the file
    if not (os.path.exists(base_dir+"cwd1.csv") and os.path.exists(base_dir+"cwd96.csv")):
        LassoPre.process(user_id)

    pred, real, train_score, test_score = LASSOFunc(base_dir+"cwd", user_id, train_start_date, train_end_date).func()

    df2 = pd.read_csv(base_dir+".csv", header=0, usecols=['time','sum'])
    # print(df2.tail())
    idxStart = df2[(df2.time == train_start_date)].index.tolist()[0]
    idxEnd = df2[(df2.time == train_end_date)].index.tolist()[0]
    scope, freq, per = LassoStatistics().statistics(df2[idxStart: idxEnd])
    print(idxStart, idxEnd)
    end_time = datetime.datetime.now()
    interval = str('%.2f' % (end_time-start_time).total_seconds())
    # params = user_id+"|"+train_start_date+"|"+train_end_date
    hist = ""
    df3 = pd.read_csv(base_dir+".csv", skiprows=idxEnd-2, nrows=3)
    dl = [df3.columns[0],df3.columns[2],df3.columns[3],df3.columns[100]]
    df3.drop(dl, axis=1, inplace=True)
    for e in df3.values:
        hist += str(e.tolist())
    print(hist)
    print(pred)
    print(real)
    msg = "LASSO|"+str(pred)+"|"+str(real)+"|"+str(hist)+"|"+scope+"|"+freq+"|"+per+"|"+start_time.strftime("%Y-%m-%d %H:%M:%S") +\
               "|"+end_time.strftime("%Y-%m-%d %H:%M:%S")+"|"+interval+"|"+user_id+"|-1"
    print(msg)
    #Sender().sendMessage(msg)
