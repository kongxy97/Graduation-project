from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from Sender import Sender
import datetime
import time
import sys

class DataExtracter(object):
    def __init__(self, path):
        self.path = path;
        self.data = pd.read_csv(self.path,encoding='utf-8');

    def select(self, range):
        return self.data.iloc[:, range];

    def getColLen(self):
        return len(self.data.columns);

    def filterDataByTime(self, date):
        t = str(datetime.datetime.strptime(date,"%Y-%m-%d").date())
        self.data.loc[:, "time"].astype('datetime64[ns]', copy=False);
        tmpdata = self.data[self.data["time"] == t];
        X = tmpdata.ix[:, '1':'93']
        return X

    def filterZerosLine(self, data):
        X = data[(data.T != 0).any()].as_matrix()
        return X

class GetImportDots:
    def __init__(self):
        pass

    def pearson_affinity(self,M):
        return 1 - np.array([[pearsonr(a, b)[0] for a in M] for b in M])


    def weight_pearson_affinity(self,M):
        return np.array([[self.weight_pearson_distance(a, b) for a in M] for b in M])

    def weight_matrix(self,s1):
        epsonRate = 0.1
        line_range = (np.max(s1)-np.min(s1)) * epsonRate
        matrix = np.ones(len(s1))
        l1 = []
        for i in range(len(s1)-1):
            if 0 < i < len(s1):
                if (s1[i] >= s1[i-1] and s1[i] > s1[i+1]) or (s1[i] > s1[i-1] and s1[i] >= s1[i+1]) \
                        or (s1[i] <= s1[i-1] and s1[i] < s1[i+1]) or (s1[i] < s1[i-1] and s1[i] <= s1[i+1]):
                    if self.tmp_distance(s1[i-1], s1[i], s1[i+1]) > line_range:
                        matrix[i] = 2.0
                    else:
                        matrix[i] = 1.5
                    l1.append(i)

        for i in range(len(l1)):
            if 0 < i < len(l1):
                if (s1[i] >= s1[i-1] and s1[i] > s1[i+1]) or (s1[i] > s1[i-1] and s1[i] >= s1[i+1]) \
                        or (s1[i] <= s1[i-1] and s1[i] < s1[i+1]) or (s1[i] < s1[i-1] and s1[i] <= s1[i+1]):
                    if self.tmp_distance(s1[i - 1], s1[i], s1[i + 1]) > line_range:
                        matrix[i] = 2.0
                    else:
                        matrix[i] = 1.5
        return matrix

    def weight_pearson_distance(self,s1, s2):
        matrix_s1 = self.weight_matrix(s1)
        matrix_s2 = self.weight_matrix(s2)
        mDLP1 = self.DLP_distance(s1, matrix_s1)
        mDLP2 = self.DLP_distance(s2, matrix_s2)
        return 1 - self.cov_DLP_distance(s1, s2, mDLP1, mDLP2, matrix_s1, matrix_s2)/\
               np.sqrt(self.cov_DLP_distance(s1, s1, mDLP1, mDLP1, matrix_s1, matrix_s1) * self.cov_DLP_distance(s2, s2, mDLP2, mDLP2, matrix_s2, matrix_s2))

    def tmp_distance(self, a, b, c):
        return abs(a+ (a-c)/2 - b)

    def DLP_distance(self, s1, matrix_s1):
        sum1 = 0.0;
        sum2 = 0.0;
        for i in range(len(s1)):
            sum1 += matrix_s1[i] * s1[i];
            sum2 += matrix_s1[i];
        return sum1/sum2;

    def cov_DLP_distance(self,s1, s2, mDLP1, mDLP2, matrix_s1, matrix_s2):
        sum1 = 0.0;
        sum2 = 0.0;
        for i in range(len(s1)):
            sum1 += (s1[i] - mDLP1) * (s2[i] -mDLP2) * matrix_s1[i] * matrix_s2[i]
            sum2 += matrix_s1[i] * matrix_s2[i];
        return sum1 / sum2;

if __name__ == "__main__":
    # before = datetime.datetime.now();
    gid = GetImportDots()
    data = DataExtracter("data/96data201501.csv");
    clusters = int(sys.argv[1])
    date = sys.argv[2]
    linkage = sys.argv[3]

    X = data.filterDataByTime(date)
    X = data.filterZerosLine(X);

    cluster = AgglomerativeClustering(n_clusters=clusters,
                                    linkage=linkage, affinity=gid.weight_pearson_affinity).fit(X)

    model = KMeans(n_clusters=clusters)
    model.fit(X)
    lastCenters_p1 = model.cluster_centers_

    str2 = "["
    result = "pearson" +"|";
    for l in range(cluster.n_clusters):
        str1 = "["
        str1 += ','.join(str('%.2f' % e) for e in np.average(X[cluster.labels_ == l].T, axis=1))
        str1 += "]"
        result += str1
        result += "#";
        str2 += str(len(X[cluster.labels_ == l])) + ","   

    str3 = "["
    for j in range(len(lastCenters_p1)):
        str1 = "["
        str1 += ','.join(str('%.2f' % e) for e in lastCenters_p1[j])
        str1 += "]"
        result += str1
        if j != len(lastCenters_p1)-1 :
            result += "#";
        else:
            result += "|";
        str3 += str(len(X[model.labels_ == j])) + ","

    result += str2[:-1] + "]#"
    result += str3[:-1] + "]|"

    result += sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]+"|"
    result += sys.argv[1]+" "+sys.argv[2]
    # after = datetime.datetime.now();
    # result += str(before.strftime("%Y-%m-%d %H:%M:%S")) + "|";
    # result += str(after.strftime("%Y-%m-%d %H:%M:%S"))+ "|";
    # interval = (after-before).total_seconds()
    # result += str('%.2f' % interval);
    # result += "|pearsonr|";
    # result += sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]+"|"
    # result += sys.argv[4]
    print(result)
    sender = Sender()
    sender.sendMessage(result)