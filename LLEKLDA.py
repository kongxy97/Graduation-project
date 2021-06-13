from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import datetime

from Sender import Sender


class LLEKLDAFunc(object):
    def __init__(self, column_x, column_y, n_components=5, kernel='sigmoid'):
        self.column_x = column_x
        self.column_y = column_y
        self.n_components = n_components
        self.kernel = kernel

    def func(self):
        model_kpca = KernelPCA(n_components=self.n_components, kernel=self.kernel)
        x_pca = model_kpca.fit_transform(self.column_x)

        model_lda = LinearDiscriminantAnalysis()
        x_lda = model_lda.fit_transform(x_pca, self.column_y)

        x_train, x_test, y_train, y_test = train_test_split(x_lda, self.column_y, random_state=24, test_size=0.25)

        logistic = LogisticRegression()
        logistic.fit(x_train, y_train)
        trs = logistic.score(x_train, y_train)
        tes = logistic.score(x_test, y_test)
        print('train score=', trs)
        print('test score=', tes)

        return "%.4f" % trs, "%.4f" % tes


if __name__ == "__main__":
    baseDir = "data/elecData/chudiandata.csv"
    # components = sys.argv[1]
    # kernel = sys.argv[2]

    components = 5
    kernel = 'sigmoid'

    startTime = datetime.datetime.now()
    df2 = pd.read_csv(baseDir,  header=None)

    columnX1 = df2[df2.columns[0:7]]
    columnY1 = df2[7]
    sc = StandardScaler()
    columnX1 = sc.fit_transform(columnX1)
    columnY = np.array(columnY1)
    columnX = np.array(columnX1)
    trs, tes = LLEKLDAFunc(columnX, columnY, n_components=components, kernel=kernel).func()

    endTime = datetime.datetime.now()
    params = "5 sigmoid"
    #params = components+" "+kernel
    interval = str('%.2f' % (endTime-startTime).total_seconds())
    startTime = startTime.strftime("%Y-%m-%d %H:%M:%S")
    endTime = endTime.strftime("%Y-%m-%d %H:%M:%S")
    msg = "LLEKLDA|"+trs+"|"+tes+"|"+startTime+"|"+endTime+"|"+interval+"|"+params+"|-1"
    print(msg)

    Sender().sendMessage(msg)
