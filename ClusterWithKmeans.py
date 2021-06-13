import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from sklearn.cluster import KMeans
from ETL.DataExtracter import DataExtracter
from MQ.Sender import Sender
import datetime
import time

class kmeansTest(object):
    def func(self):
        before = datetime.datetime.now();
        data = DataExtracter("data/96data201501.csv");
        clusters = int(sys.argv[1])
        date = sys.argv[2]
        X = data.filterDataByTime(date);
        X = data.filterZerosLine(X);

        model = KMeans(n_clusters=clusters)

        model.fit(X)
        lastCenters_p1 = model.cluster_centers_
        result = str(clusters) +"|";
        str2 = "["
        for j in range(len(lastCenters_p1)):
            str1 = "["
            str1 += ','.join(str('%.2f' % e) for e in lastCenters_p1[j])
            str1 += "]"
            result += str1
            result += "|";
            str2 += str(len(X[model.labels_ == j])) + ","

        result += str2[:-1] + "]|"
        # result += ','.join(str(len(X[model.labels_ == j])) for j in range(len(lastCenters_p1)))
        # result += "]|";

        after = datetime.datetime.now();
        result += str(before.strftime("%Y-%m-%d %H:%M:%S")) + "|";
        result += str(after.strftime("%Y-%m-%d %H:%M:%S"))+ "|";
        interval = (after-before).total_seconds()

        result += str('%.2f' % interval)
        result += "|kmeans|";
        result += sys.argv[1]+" "+sys.argv[2]+"|"
        result += sys.argv[3]
        print(result)
        Sender.sendMessage(result)




if __name__ == "__main__":
    tt = kmeansTest();
    tt.func()
