# -*- coding:utf-8 -*-
# Time : 2021/6/20
# Author : Xiangyuan Kong
import datetime
import random
import sys

import numpy as np
from matplotlib import pyplot
from sklearn.metrics.cluster import silhouette_score

from ETL.Extract import DataExtracter
from ETL.Transform import Transformer
from numpy import seterr

seterr(all='raise')


class KMeans(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    """
    centers : 本次聚类中心 {num1: center1, num2: center2}
    clf_ : 聚类分组 {0: [array1, array2, ...], 1: [array1, array2, ...], ...}
    use_evo : 是否使用演化聚类
    prev_time_centers : t-1时刻的聚类中心
    prev_data : t-1时刻的数据集
    prev_quality : t-1时刻聚类质量
    """

    def __init__(self, k=2, tolerance=0.0001, max_iter=300, use_evo=False, prev_time_centers=None, prev_data=None,
                 prev_quality=None, cp=0.2, beta=0.2):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers_ = None
        self.clf_ = {}
        self.use_evo = use_evo
        self.prev_time_centers = prev_time_centers
        self.prev_data = prev_data
        self.prev_quality = prev_quality
        self.cp = cp
        self.beta = beta
        self.cur_quality = None
        self.reset = False

    def fit(self, data):
        self.centers_ = {}
        if self.use_evo:
            self.centers_ = self.prev_time_centers
        else:
            for i in range(self.k_):
                self.centers_[i] = []
                for j in range(24):
                    self.centers_[i].append(random.uniform(0, 5.0))

        for i in range(self.max_iter_):
            self.clf_ = {}
            for j in range(self.k_):
                self.clf_[j] = []
            # print("质点:", self.centers_)
            for feature in data:
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    # distances.append(np.linalg.norm(feature - self.centers_[center]))
                    ora = np.linalg.norm(feature - self.centers_[center])
                    num = np.array(feature).dot(np.array(self.centers_[center]))
                    de_nom = np.linalg.norm(feature) * np.linalg.norm(self.centers_[center])
                    if num == 0 and de_nom == 0:
                        de_nom += 0.01
                    cos = num / de_nom
                    sim = 0.5 - 0.5 * cos
                    distances.append(0.9 * sim + 0.02 * ora)
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:", self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                # print(self.clf_[c])
                # if not self.clf_[c]:
                #     self.clf_[c] = self.clf_[c-1]
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break
        if self.use_evo:
            # 使用演化聚类公式更新聚类中心
            temp_center = self.centers_
            for num, center in self.centers_.items():
                prev_num, closest_center = self.closest(center)
                gamma = len(self.clf_[num]) / (len(self.clf_[num]) + len(self.prev_data[prev_num]))
                center = (1 - gamma) * self.cp * closest_center + gamma * (1 - self.cp) * center
                self.centers_[num] = center
            sq = self.sq()
            hq = self.hq(data)
            self.cur_quality = sq - self.cp * hq
            if (self.prev_quality - self.cur_quality) / self.prev_quality < self.beta:
                # 不使用演化聚类
                print("Here!")
                self.centers_ = temp_center
                self.reset = True
            else:
                self.cur_quality = sq
        else:
            self.cur_quality = self.sq()

    def predict(self, p_data):
        distances = [np.linalg.norm(np.array(p_data) - np.array(self.centers_[center])) for center in self.centers_]
        index = distances.index(min(distances))
        return index

    def sq(self):
        label = []
        data_list = []
        for num, data_set in self.clf_.items():
            for data in data_set:
                label.append(num)
                data_list.append(data)
        return silhouette_score(data_list, labels=label)

    def hq(self, x_data):
        label = []
        data_list = []
        for x_data_item in x_data:
            min_label = -1
            min_dis = sys.maxsize
            for num, center in self.prev_time_centers.items():
                cur_distance = np.sqrt(np.sum((np.array(x_data_item) - np.array(center)) ** 2))
                if cur_distance < min_dis:
                    min_dis = cur_distance
                    min_label = num
            label.append(min_label)
            data_list.append(x_data_item)
        return silhouette_score(data_list, labels=label)

    def closest(self, center):
        # 返回t-1时刻距离最近的聚类中心
        min_distance = sys.maxsize
        closest_center = []
        min_num = -1
        for last_num, last_center in self.prev_time_centers.items():
            cur_distance = np.sqrt(np.sum((np.array(last_center) - np.array(center)) ** 2))
            if cur_distance < min_distance:
                closest_center = last_center
                min_num = last_num
        return min_num, closest_center


def main():
    extractor = DataExtracter("../data/96data201501.csv")
    rang = range(0, extractor.getColLen() - 3)
    transformer = Transformer(extractor.select(rang))
    date = "2015-01-01"
    cur_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    last_center = {}
    last_data = {}
    last_quality = 0
    for i in range(2):
        time = str(cur_date + datetime.timedelta(days=i))
        cur_data = transformer.filterTime(time)
        data_matrix = cur_data.loc[:, '1':'93']  # 上述操作主要是数据的加载和抽取
        if i == 0:
            model = KMeans(k=6)
        else:
            model = KMeans(k=6, use_evo=True, prev_time_centers=last_center, prev_data=last_data,
                           prev_quality=last_quality)
        model.fit(data_matrix.values)
        last_quality = model.cur_quality
        last_center = model.centers_
        # do something on model.centers_
        pyplot.figure(100)
        # print(last_center)
        if model.reset:
            print(time + " reset")
        for num, centers in last_center.items():
            pyplot.plot(centers, label=num)
        last_data = model.clf_
        # do something on model.clf_
        for num, data_points in last_data.items():
            pyplot.figure(num)
            for data in data_points:
                pyplot.plot(data)
        pyplot.show()


if __name__ == '__main__':
    """
    96data201501.csv数据格式：
            y_cid        time      1       5  ...    93      sum  max_load min_load
    0      5776593  2015-01-01  0.070  0.0600  ...  0.06   1.8334     0.14     0.06
    32     5782689  2015-01-01  0.650  0.6200  ...  1.27  22.6817     2.10     0.10
    71     5880741  2015-01-01  0.010  0.0000  ...  0.01   0.0662     0.01     0.00
    148    5880321  2015-01-01  1.300  0.3400  ...  1.18   9.8400     1.30     0.06
    181    5881086  2015-01-01  0.000  0.0000  ...  0.00   0.0000     0.00     0.00
    ...        ...         ...    ...     ...  ...   ...      ...      ...      ...
    33832  5928173  2015-01-01  0.220  0.1500  ...  0.14   4.6200     0.83     0.01
    33864  5928185  2015-01-01  0.000  0.0000  ...  0.00   0.0000     0.00     0.00
    33896  5928299  2015-01-01  0.000  0.0000  ...  0.00   0.0000     0.00     0.00
    33928  5928320  2015-01-01  0.101  0.0728  ...  0.36   4.1431     1.06     0.00
    33960  5928440  2015-01-01  0.000  0.0000  ...  0.00   0.0000     0.00     0.00
    
    处理后数据格式：
    [1016 rows x 29 columns]
    [[0.07   0.06   0.08   ... 0.07   0.08   0.06  ]
     [0.65   0.62   0.64   ... 1.1145 1.45   1.27  ]
     [0.01   0.     0.     ... 0.     0.     0.01  ]
     ...
     [0.     0.     0.     ... 0.     0.     0.    ]
     [0.101  0.0728 0.0593 ... 0.75   1.06   0.36  ]
     [0.     0.     0.     ... 0.     0.     0.    ]]
    """
    main()
    # print(lastData)
    # print(data_matrix.values)
    # print("---------------------------------------------------")
    # x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    # k_means = KMeans(k=2)
    # k_means.fit(x)
    # print(k_means.centers_)
    # for clustering_center in k_means.centers_:
    #     pyplot.scatter(k_means.centers_[clustering_center][0], k_means.centers_[clustering_center][1], marker='*',
    #                    s=150)
    #
    # for cat in k_means.clf_:
    #     for point in k_means.clf_[cat]:
    #         pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))
    #
    # predict = [[2, 1], [6, 9]]
    # for data_point in predict:
    #     cat = k_means.predict(predict)
    #     pyplot.scatter(data_point[0], data_point[1], c=('r' if cat == 0 else 'b'), marker='x')
    #
    # pyplot.show()
