# -*- coding:utf-8 -*-
# Time : 2021/6/20
# Author : Xiangyuan Kong

import numpy as np
from matplotlib import pyplot


class KMeans(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300, use_evo=False, prev_time_centers=None, prev_data=None,
                 prev_quality=None, t=3, cp=0.2, beta=0.2):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers_ = None
        self.clf_ = {}
        self.use_evo = use_evo
        self.prev_time_centers = prev_time_centers
        self.prev_data = prev_data
        self.prev_quality = prev_quality
        self.t = t
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
                self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for j in range(self.k_):
                self.clf_[j] = []
            print("质点:", self.centers_)
            for feature in data:
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            print("分组情况:", self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
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
            # todo 使用演化聚类公式更新聚类中心
            temp_center = self.centers_
            for num, center in self.centers_:
                prev_num, closest_center = self.closest(center)
                gamma = len(self.clf_[num]) / (len(self.clf_[num]) + len(self.prev_data[prev_num]))
                center = (1 - gamma) * self.cp * closest_center + gamma * (1 - self.cp) * center
                self.centers_[num] = center
            sq = self.sq()
            hq = self.hq()
            self.cur_quality = sq - self.cp * hq
            if (self.prev_quality - self.cur_quality) / self.prev_quality < self.beta:
                # 不使用演化聚类
                self.centers_ = temp_center
                self.reset = True
            else:
                self.cur_quality = sq

    def predict(self, p_data):
        distances = [np.linalg.norm(np.array(p_data) - np.array(self.centers_[center])) for center in self.centers_]
        index = distances.index(min(distances))
        return index

    def sq(self):
        # todo
        return self.cp

    def hq(self):
        # todo
        return self.cp

    def closest(self, center):
        # todo 返回t-1时刻距离最近的聚类中心
        return self.centers_[0], (self.centers_ - center)[1]


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = KMeans(k=2)
    k_means.fit(x)
    print(k_means.centers_)
    for clustering_center in k_means.centers_:
        pyplot.scatter(k_means.centers_[clustering_center][0], k_means.centers_[clustering_center][1], marker='*',
                       s=150)

    for cat in k_means.clf_:
        for point in k_means.clf_[cat]:
            pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))

    # predict = [[2, 1], [6, 9]]
    # for data_point in predict:
    #     cat = k_means.predict(predict)
    #     pyplot.scatter(data_point[0], data_point[1], c=('r' if cat == 0 else 'b'), marker='x')

    pyplot.show()
