# -*- coding: utf8 -*-
"""
Created on 2016年12月5日

@author: maitian13
"""
import matplotlib.pyplot as pl

# def getRelatedCenters(lastCenters,currCenters):
import datetime
import sys
import matplotlib
import pandas as pd
from sklearn.cluster import KMeans as OriginKMeans
from sklearn.metrics.cluster import silhouette_score as SS
import numpy as np
import sklearn.cluster.k_means_

# import time
from ETL.Extract import DataExtracter
from ETL.Transform import Transformer
from Evolutionary_Kmeans import KMeans
from HQ import HistoryQ

matplotlib.use("Agg")


class kmeansT(object):
    #
    #
    # test0
    # 使用普通的聚类算法，分别对2015-01-01和2015-01-02两天的用电数据进行聚类，然后将聚类中心画出来
    # 画图的目的是想表明使用普通的聚类算法得到的两个聚类中心差异巨大
    #
    def test0(self):
        extracter = DataExtracter("data/96data201501.csv")
        rang = range(0, extracter.getColLen() - 3)
        transformer = Transformer(extracter.select(rang))
        # data_matrix=transformer.to_maxtrix()
        clusters = int(sys.argv[1])
        date = sys.argv[2]
        t = str(datetime.datetime.strptime(date, "%Y-%m-%d").date())
        lastData = transformer.filterTime(t)
        data_matrix = lastData.ix[:, '1':'93'].as_matrix()  # 上述操作主要是数据的加载和抽取
        # for i in range(2,10):
        model = KMeans(n_clusters=clusters)  # 初始化

        # 画2015-01-01的聚类结果
        lastpred_p1 = model.fit_predict(data_matrix, "arg1", "arg2", isFirst=True)  # 当设置isFirst=True时，本次聚类将不会使用演进聚类
        lastCenters_p1 = model.cluster_centers_
        pl.figure(figsize=(12, 6))
        pl.title(t)
        pl.subplot(1, 2, 1)
        newdata = pd.DataFrame()
        for j in range(len(lastCenters_p1)):
            pl.plot(range(1, 25), lastCenters_p1[j], label='C' + str(j + 1))
            newdata['C' + str(j + 1)] = lastCenters_p1[j]
        pl.legend(loc="upper middle", ncol=clusters, borderaxespad=0., fontsize=8)  # 同上

        print("---------")

        # 画2015-01-02的聚类结果
        pl.subplot(1, 2, 2)
        before = datetime.datetime.strptime(date, "%Y-%m-%d")
        lastData = transformer.filterTime(str((before + datetime.timedelta(days=1)).date()))
        data_matrix = lastData.ix[:, '1':'93'].as_matrix()
        lastpred_p1 = model.fit_predict(data_matrix, "arg1", "arg2", isFirst=True)
        lastCenters_p1 = model.cluster_centers_
        #         print lastCenters_p1
        for j in range(len(lastCenters_p1)):
            pl.plot(range(1, 25), lastCenters_p1[j], label='C' + str(j + 1))
            newdata['B' + str(j + 1)] = lastCenters_p1[j]
        pl.legend(loc="upper middle", ncol=clusters, borderaxespad=0., fontsize=8)
        pl.show()

    #
    #
    # test01
    # 使用演进的聚类算法，分别对2015-01-01和2015-01-02两天的用电数据进行聚类，然后将聚类中心画出来
    # 画图的目的是想表明使用演进的聚类算法得到的两个聚类中心差异小，相比test0得到的聚类中心更优良
    #
    def test01(self):
        extracter = DataExtracter("data/96data201501.csv")
        rang = range(0, extracter.getColLen() - 3)
        transformer = Transformer(extracter.select(rang))
        date = sys.argv[2]
        # data_matrix=transformer.to_maxtrix()
        t = str(datetime.datetime.strptime(date, "%Y-%m-%d").date())


        lastData = transformer.filterTime(t)
        data_matrix = lastData.ix[:, '1':'93'].as_matrix()  # 上述操作主要是数据的加载和抽取
        clusters = int(sys.argv[1])  # 初始化聚类的类别数
        model = KMeans(n_clusters=clusters)
        result = "evo" + "|"
        str2 = "["

        # 画2015-01-01的聚类中心
        lastpred_p1 = model.fit_predict(data_matrix, "arg1", "arg2", isFirst=True)  # 当设置isFirst=True时，本次聚类将不会使用演进聚类
        lastCenters_p1 = model.cluster_centers_
        pl.figure(figsize=(12, 6))
        pl.title(t)
        pl.subplot(120 + 1)
        for j in range(len(lastCenters_p1)):
            str1 = "["
            str1 += ','.join(str(e) for e in lastCenters_p1[j])
            str1 += "]"
            result += str1
            result += "#"
            str2 += str(len(data_matrix[model.labels_ == j])) + ","
            pl.plot(range(1, 25), lastCenters_p1[j], label='C' + str(j + 1))
        pl.legend(loc="upper middle", ncol=clusters, borderaxespad=0., fontsize=8)

        result += str2[:-1] + "]|"

        # 画2015-01-02的聚类中心，这里使用了演进聚类
        pl.subplot(120 + 2)
        str2 = "["
        before = datetime.datetime.strptime(date, "%Y-%m-%d")
        lastData = transformer.filterTime(str((before + datetime.timedelta(days=1)).date()))
        data_matrix = lastData.ix[:, '1':'93'].as_matrix()
        lastpred_p1 = model.fit_predict(data_matrix, lastCenters=lastCenters_p1.copy(), lastCategory="hehe",
                                        isFirst=False)  # 当设置isFirst=False时，本次聚类将会使用演进聚类，同时需要提供上一次的聚类结果lastCenters
        print "****************************"
        print np.array(lastData)
        print np.array(lastpred_p1)
        print "****************************"
        lastCenters_p1 = model.cluster_centers_
        for j in range(len(lastCenters_p1)):
            str1 = "["
            str1 += ','.join(str(e) for e in lastCenters_p1[j])
            str1 += "]"
            result += str1
            if j != len(lastCenters_p1) - 1:
                result += "#"
            else:
                result += "|"
            str2 += str(len(data_matrix[model.labels_ == j])) + ","
            pl.plot(range(1, 25), lastCenters_p1[j], label='C' + str(j + 1))
        pl.legend(loc="upper middle", ncol=clusters, borderaxespad=0., fontsize=8)
        pl.show()
        result += str2[:-1] + "]"
        return result

    #
    #
    # test21
    # 论文里面beta参数的变化对sq\hq质量的影响
    # use
    def test21(self):
        extracter = DataExtracter("data/96data2016.csv")
        transformer = Transformer(extracter.data)
        date = sys.argv[2]
        t = str(datetime.datetime.strptime(date, "%Y-%m-%d").date())
        before = datetime.datetime.strptime(date, "%Y-%m-%d")
        lastData = transformer.filterTime(t)
        data_matrix = lastData.ix[:, '1':'93'].as_matrix().copy()
        clusters = int(sys.argv[1])
        deltaDays = int(sys.argv[3])
        beta = [0, 0.125, 0.5]
        lines = ['mo:', "cx-", "D-.", "kp--", "h:"]
        pl.title("Average HQ")
        newdata = pd.DataFrame()
        str2 = ""
        for k in range(0, len(beta)):
            str1 = "["
            model = KMeans(n_clusters=clusters)
            model1 = OriginKMeans(n_clusters=clusters)
            lastpred_p1 = model.fit_predict(data_matrix, "arg1", "arg2", isFirst=True)
            lastCenters_p1 = model.cluster_centers_
            lastpred_p2 = lastpred_p1.copy()
            lastCenters_p2 = lastCenters_p1.copy()
            lastpred_p3 = lastpred_p1.copy()
            lastCenters_p3 = lastCenters_p1.copy()

            res = []
            sq = []
            sq1 = []
            sq2 = []
            hq = []
            hq1 = []
            hq2 = []
            sqsum = 0
            sqsum1 = 0
            sqsum2 = 0
            hqsum = 0
            hqsum1 = 0
            hqsum2 = 0
            pcq = []
            pcq1 = []
            pcq2 = []
            cumu = []
            cumu1 = []
            for i in range(1, deltaDays):
                t1 = before + datetime.timedelta(days=i)
                print("------------------------", i, t1.strftime("%Y-%m-%d"))

                currData = transformer.filterTime(t1.strftime("%Y-%m-%d"))
                data_matrix1 = currData.ix[:, '1':'93'].as_matrix().copy()
                currpred_p1 = model.fit_predict(data_matrix1, lastCenters=lastCenters_p1.copy(), lastCategory="hehe",
                                                isFirst=True)
                tmp_p1 = model.predict(data_matrix)
                currCenters_p1 = model.cluster_centers_
                currpred_p2 = model.fit_predict(data_matrix1, lastCenters=lastCenters_p2.copy(), lastCategory="hehe",
                                                isFirst=False)
                tmp_p2 = model.predict(data_matrix)
                currCenters_p2 = model.cluster_centers_
                currpred_p3 = model.fit_predict(data_matrix1, lastCenters=lastCenters_p3.copy(), lastCategory="hehe",
                                                isFirst=False)
                tmp_p3 = model.predict(data_matrix)
                currCenters_p3 = model.cluster_centers_
                tmp = SS(data_matrix1, currpred_p1, metric="cosine")
                tmp1 = SS(data_matrix1, currpred_p3, metric="cosine")
                print(abs((tmp - tmp1) / tmp1))

                if tmp > tmp1 and abs((tmp - tmp1) / tmp1) > beta[k]:
                    print("here")

                    currpred_p3 = currpred_p1.copy()
                    tmp_p3 = tmp_p1.copy()
                    currCenters_p3 = currCenters_p1.copy()
                history = HistoryQ(lastCenters_p1, currCenters_p1, lastpred_p1, currpred_p1, clusters)
                history1 = HistoryQ(lastCenters_p2, currCenters_p2, lastpred_p2, currpred_p2, clusters)
                history2 = HistoryQ(lastCenters_p3, currCenters_p3, lastpred_p3, currpred_p3, clusters)
                sqsum += SS(data_matrix1, currpred_p1, metric="cosine")
                sqsum1 += SS(data_matrix1, currpred_p2, metric="cosine")
                sqsum2 += SS(data_matrix1, currpred_p3, metric="cosine")
                hqsum += -history.calc2()
                hqsum1 += -history1.calc2()
                hqsum2 += -history2.calc2()

                sq.append(0.3 + sqsum / i)
                sq1.append(0.3 + sqsum1 / i)
                sq2.append(0.3 + sqsum2 / i)

                hq.append(hqsum / i)
                hq1.append(hqsum1 / i)
                hq2.append(hqsum2 / i)

                pcq.append(SS(data_matrix, tmp_p1, metric="cosine"))
                pcq1.append(SS(data_matrix, tmp_p2, metric="cosine"))
                pcq2.append(SS(data_matrix, tmp_p3, metric="cosine"))
                cumu.append((sum(sq) - sum(sq1)) / len(sq))
                cumu1.append((sum(sq) - sum(sq2)) / len(sq))
                currData['pred'] = currpred_p1
                currData['class'] = 0

                for j in range(len(currCenters_p1)):
                    currData.loc[currData['pred'] == j, 'class'] = tmp

                res.append(currData.ix[:, ['ycid', 'time', 'class']])

                lastCenters_p1 = currCenters_p1
                lastpred_p1 = currpred_p1
                lastCenters_p2 = currCenters_p2
                lastpred_p2 = currpred_p2
                lastCenters_p3 = currCenters_p3
                lastpred_p3 = currpred_p3
                data_matrix = data_matrix1.copy()

            res = pd.concat(res)

            print("sq_sum_original:", sum(sq) / len(sq))

            print("sq_sum_optimal:", sum(sq1) / len(sq))

            print("sq_sum_final:", sum(sq2) / len(sq))

            print("hq_sum_original:", sum(hq) / len(sq))

            print("hq_sum_optimal:", sum(hq1) / len(sq))

            print("hq_sum_final:", sum(hq2) / len(sq))
            pl.plot(range(1, deltaDays), sq2, lines[k], label=r'$\beta$=' + str(beta[k]))
            str1 += ','.join(str(e) for e in hq2)
            str1 += "]"
            str2 += str1
            str2 += "|"
        pl.legend(loc="upper right", ncol=clusters, borderaxespad=0., fontsize=10)

        pl.show()
        return str2

    #
    #
    # test31
    # 论文里面比较Kmeans、Evolutionary、Optimal三种算法的hq和sq质量随着时间的变化
    #
    def test31(self):
        extracter = DataExtracter("data/96data2016.csv")
        transformer = Transformer(extracter.data)
        date = sys.argv[2]
        t = str(datetime.datetime.strptime(date, "%Y-%m-%d").date())
        before = datetime.datetime.strptime(date, "%Y-%m-%d")
        lastData = transformer.filterTime(t)
        data_matrix = lastData.ix[:, '1':'93'].as_matrix().copy()
        clusters = int(sys.argv[1])
        deltaDays = int(sys.argv[3]) + 1
        beta = [0.125]
        lines = ['mo:', "cx-", "D-.", "kp--", "h:"]
        pl.title("Average HQ")
        str2 = "["
        for k in range(0, len(beta)):
            model = KMeans(n_clusters=clusters)
            model1 = OriginKMeans(n_clusters=clusters)
            lastpred_p1 = model.fit_predict(data_matrix, "arg1", "arg2", isFirst=True)
            lastCenters_p1 = model.cluster_centers_
            lastpred_p2 = lastpred_p1.copy()
            lastCenters_p2 = lastCenters_p1.copy()
            lastpred_p3 = lastpred_p1.copy()
            lastCenters_p3 = lastCenters_p1.copy()
            res = []
            sq = []
            sq1 = []
            sq2 = []
            hq = []
            hq1 = []
            hq2 = []
            sqsum = 0
            sqsum1 = 0
            sqsum2 = 0
            hqsum = 0
            hqsum1 = 0
            hqsum2 = 0
            pcq = []
            pcq1 = []
            pcq2 = []
            cumu = []
            cumu1 = []
            for i in range(1, deltaDays):
                t1 = before + datetime.timedelta(days=i)
                print("------------------------", i, t1.strftime("%Y-%m-%d"))

                currData = transformer.filterTime(t1.strftime("%Y-%m-%d"))
                data_matrix1 = currData.ix[:, '1':'93'].as_matrix().copy()
                currpred_p1 = model.fit_predict(data_matrix1, lastCenters=lastCenters_p1.copy(), lastCategory="hehe",
                                                isFirst=True)
                tmp_p1 = model.predict(data_matrix)
                currCenters_p1 = model.cluster_centers_
                currpred_p2 = model.fit_predict(data_matrix1, lastCenters=lastCenters_p2.copy(), lastCategory="hehe",
                                                isFirst=False)
                tmp_p2 = model.predict(data_matrix)
                currCenters_p2 = model.cluster_centers_
                currpred_p3 = model.fit_predict(data_matrix1, lastCenters=lastCenters_p3.copy(), lastCategory="hehe",
                                                isFirst=False)
                tmp_p3 = model.predict(data_matrix)
                currCenters_p3 = model.cluster_centers_
                tmp = SS(data_matrix1, currpred_p1, metric="cosine")
                tmp1 = SS(data_matrix1, currpred_p3, metric="cosine")
                print(abs((tmp - tmp1) / tmp1))

                if tmp > tmp1 and abs((tmp - tmp1) / tmp1) > beta[k]:
                    print("here")

                    currpred_p3 = currpred_p1.copy()
                    tmp_p3 = tmp_p1.copy()
                    currCenters_p3 = currCenters_p1.copy()
                history = HistoryQ(lastCenters_p1, currCenters_p1, lastpred_p1, currpred_p1, clusters)
                history1 = HistoryQ(lastCenters_p2, currCenters_p2, lastpred_p2, currpred_p2, clusters)
                history2 = HistoryQ(lastCenters_p3, currCenters_p3, lastpred_p3, currpred_p3, clusters)
                sqsum += SS(data_matrix1, currpred_p1, metric="cosine")
                sqsum1 += SS(data_matrix1, currpred_p2, metric="cosine")
                sqsum2 += SS(data_matrix1, currpred_p3, metric="cosine")
                hqsum += -history.calc2()
                hqsum1 += -history1.calc2()
                hqsum2 += -history2.calc2()
                sq.append(0.3 + SS(data_matrix1, currpred_p1, metric="cosine"))
                sq1.append(0.3 + SS(data_matrix1, currpred_p2, metric="cosine"))
                sq2.append(0.3 + SS(data_matrix1, currpred_p3, metric="cosine"))
                hq.append(-history.calc2())
                hq1.append(-history1.calc2())
                hq2.append(-history2.calc2())

                pcq.append(SS(data_matrix, tmp_p1, metric="cosine"))
                pcq1.append(SS(data_matrix, tmp_p2, metric="cosine"))
                pcq2.append(SS(data_matrix, tmp_p3, metric="cosine"))
                cumu.append((sum(sq) - sum(sq1)) / len(sq))
                cumu1.append((sum(sq) - sum(sq2)) / len(sq))
                currData['pred'] = currpred_p1
                currData['class'] = 0
                # pl.subplot(220+i)
                # pl.title(t1.strftime("%Y-%m-%d"))
                for j in range(len(currCenters_p1)):
                    currData.loc[currData['pred'] == j, 'class'] = tmp
                    # pl.plot(range(1,25),currCenters_p2[j],label=tmp)
                res.append(currData.ix[:, ['ycid', 'time', 'class']])
                # pl.legend(loc="upper middle",ncol=clusters,borderaxespad=0.,fontsize=8)
                lastCenters_p1 = currCenters_p1
                lastpred_p1 = currpred_p1
                lastCenters_p2 = currCenters_p2
                lastpred_p2 = currpred_p2
                lastCenters_p3 = currCenters_p3
                lastpred_p3 = currpred_p3
                data_matrix = data_matrix1.copy()
            res = pd.concat(res)
            # res.to_csv("../data/user/test/2015_v1.csv",index=False)
            print("sq_sum_original:", sum(sq) / len(sq))

            print("sq_sum_optimal:", sum(sq1) / len(sq))

            print("sq_sum_final:", sum(sq2) / len(sq))

            print("hq_sum_original:", sum(hq) / len(sq))

            print("hq_sum_optimal:", sum(hq1) / len(sq))

            print("hq_sum_final:", sum(hq2) / len(sq))

            # pl.plot(range(1,deltaDays),sq,'cx-',label="Kmeans")
            # pl.plot(range(1,deltaDays),sq1,'b-.',label="Evolutionary")
            # pl.plot(range(1,deltaDays),hq2,lines[k],label="cp="+bytes(beta[k]))
            # pl.plot(range(1,deltaDays),sq2,'D-.',label="Optimal")

        pl.title("HQ")
        pl.plot(range(1,deltaDays),hq,'cx-',label="Kmeans")
        str2 += ','.join(str(e) for e in hq)
        str2 += "]#["
        pl.plot(range(1,deltaDays),hq1,'mo:',label="Evolutionary")
        str2 += ','.join(str(e) for e in hq1)
        str2 += "]#["
        pl.plot(range(1,deltaDays),hq2,'D-.',label="Optimal")
        str2 += ','.join(str(e) for e in hq2)
        str2 += "]#["
        pl.legend(loc="upper middle",ncol=clusters,borderaxespad=0.,fontsize=10)
        pl.show()

        pl.subplot(210+2)
        pl.title("SQ")
        pl.plot(range(1,deltaDays),sq,'cx-',label="Kmeans")
        str2 += ','.join(str(e) for e in sq)
        str2 += "]#["
        pl.plot(range(1,deltaDays),sq1,'mo:',label="Evolutionary")
        str2 += ','.join(str(e) for e in sq1)
        str2 += "]#["
        pl.plot(range(1,deltaDays),sq2,'D-',label="Optimal")
        str2 += ','.join(str(e) for e in sq2)
        str2 += "]|"
        print(hq)

        print(hq1)

        print(hq2)

        print(sq)

        print(sq1)

        print(sq2)

        newdata = pd.DataFrame()
        newdata['hq'] = hq
        newdata['hq1'] = hq1
        newdata['hq2'] = hq2
        newdata['sq'] = sq
        newdata['sq1'] = sq1
        newdata['sq2'] = sq2
        pl.legend(loc="upper middle",ncol=clusters,borderaxespad=0.,fontsize=10)
        # pl.legend(loc="upper middle",ncol=clusters,borderaxespad=0.,fontsize=10)
        pl.show()
        str2 += sys.argv[1] + " " + sys.argv[2] + " " + sys.argv[3] + "|"
        str2 += sys.argv[1] + " " + sys.argv[2]
        return str2

    #
    #
    # test41
    # 论文里面比较Kmeans、Evolutionary、Optimal三种算法的sq质量变化
    #
    def test41(self):
        extracter = DataExtracter("data/96data2016.csv")
        transformer = Transformer(extracter.data)
        date = sys.argv[2]
        t = str(datetime.datetime.strptime(date, "%Y-%m-%d").date())
        before = datetime.datetime.strptime(date, "%Y-%m-%d")
        lastData = transformer.filterTime(t)
        data_matrix = lastData.ix[:, '1':'93'].as_matrix().copy()
        clusters = int(sys.argv[1])
        deltaDays = int(sys.argv[3])
        beta = [0.125]
        lines = ['mo:', "cx-", "D-.", "kp--", "h:"]
        pl.title("Average SQ")
        newdata = pd.DataFrame()
        for k in range(0, len(beta)):
            model = KMeans(n_clusters=clusters)
            model1 = OriginKMeans(n_clusters=clusters)
            lastpred_p1 = model.fit_predict(data_matrix, "arg1", "arg2", isFirst=True)
            lastCenters_p1 = model.cluster_centers_
            lastpred_p2 = lastpred_p1.copy()
            lastCenters_p2 = lastCenters_p1.copy()
            lastpred_p3 = lastpred_p1.copy()
            lastCenters_p3 = lastCenters_p1.copy()
            res = []
            sq = []
            sq1 = []
            sq2 = []
            hq = []
            hq1 = []
            hq2 = []
            sqsum = 0
            sqsum1 = 0
            sqsum2 = 0
            hqsum = 0
            hqsum1 = 0
            hqsum2 = 0
            pcq = []
            pcq1 = []
            pcq2 = []
            cumu = []
            cumu1 = []
            for i in range(1, deltaDays):
                t1 = before + datetime.timedelta(days=i)
                print("------------------------", i, t1.strftime("%Y-%m-%d"))

                currData = transformer.filterTime(t1.strftime("%Y-%m-%d"))
                data_matrix1 = currData.ix[:, '1':'93'].as_matrix().copy()
                currpred_p1 = model.fit_predict(data_matrix1, lastCenters=lastCenters_p1.copy(), lastCategory="hehe",
                                                isFirst=True)
                tmp_p1 = model.predict(data_matrix)
                currCenters_p1 = model.cluster_centers_
                currpred_p2 = model.fit_predict(data_matrix1, lastCenters=lastCenters_p2.copy(), lastCategory="hehe",
                                                isFirst=False)
                tmp_p2 = model.predict(data_matrix)
                currCenters_p2 = model.cluster_centers_
                currpred_p3 = model.fit_predict(data_matrix1, lastCenters=lastCenters_p3.copy(), lastCategory="hehe",
                                                isFirst=False)
                tmp_p3 = model.predict(data_matrix)
                currCenters_p3 = model.cluster_centers_
                tmp = SS(data_matrix1, currpred_p1, metric="cosine")
                tmp1 = SS(data_matrix1, currpred_p3, metric="cosine")
                print(abs((tmp - tmp1) / tmp1))

                if tmp > tmp1 and abs((tmp - tmp1) / tmp1) > beta[k]:
                    print("here")

                    currpred_p3 = currpred_p1.copy()
                    tmp_p3 = tmp_p1.copy()
                    currCenters_p3 = currCenters_p1.copy()
                history = HistoryQ(lastCenters_p1, currCenters_p1, lastpred_p1, currpred_p1, clusters)
                history1 = HistoryQ(lastCenters_p2, currCenters_p2, lastpred_p2, currpred_p2, clusters)
                history2 = HistoryQ(lastCenters_p3, currCenters_p3, lastpred_p3, currpred_p3, clusters)
                sqsum += SS(data_matrix1, currpred_p1, metric="cosine")
                sqsum1 += SS(data_matrix1, currpred_p2, metric="cosine")
                sqsum2 += SS(data_matrix1, currpred_p3, metric="cosine")
                hqsum += -history.calc2()
                hqsum1 += -history1.calc2()
                hqsum2 += -history2.calc2()
                sq.append(0.3 + sqsum / i)
                sq1.append(0.3 + sqsum1 / i)
                sq2.append(0.3 + sqsum2 / i)

                hq.append(hqsum / i)
                hq1.append(hqsum1 / i)
                hq2.append(hqsum2 / i)

                pcq.append(SS(data_matrix, tmp_p1, metric="cosine"))
                pcq1.append(SS(data_matrix, tmp_p2, metric="cosine"))
                pcq2.append(SS(data_matrix, tmp_p3, metric="cosine"))
                cumu.append((sum(sq) - sum(sq1)) / len(sq))
                cumu1.append((sum(sq) - sum(sq2)) / len(sq))
                currData['pred'] = currpred_p1
                currData['class'] = 0
                for j in range(len(currCenters_p1)):
                    currData.loc[currData['pred'] == j, 'class'] = tmp
                    # pl.plot(range(1,25),currCenters_p2[j],label=tmp)
                res.append(currData.ix[:, ['ycid', 'time', 'class']])
                # pl.legend(loc="upper middle",ncol=clusters,borderaxespad=0.,fontsize=8)
                lastCenters_p1 = currCenters_p1
                lastpred_p1 = currpred_p1
                lastCenters_p2 = currCenters_p2
                lastpred_p2 = currpred_p2
                lastCenters_p3 = currCenters_p3
                lastpred_p3 = currpred_p3
                data_matrix = data_matrix1.copy()
            res = pd.concat(res)
            print("sq_sum_original:", sum(sq) / len(sq))

            print("sq_sum_optimal:", sum(sq1) / len(sq))

            print("sq_sum_final:", sum(sq2) / len(sq))

            print("hq_sum_original:", sum(hq) / len(sq))

            print("hq_sum_optimal:", sum(hq1) / len(sq))

            print("hq_sum_final:", sum(hq2) / len(sq))

            pl.plot(range(1, deltaDays), sq, 'cx-', label="Kmeans")
            pl.plot(range(1, deltaDays), sq1, 'mo:', label="Evolutionary")
            # pl.plot(range(1,deltaDays),hq2,lines[k],label="cp="+bytes(beta[k]))
            pl.plot(range(1, deltaDays), sq2, 'D-.', label="Optimal")
            str2 = "["
            str2 += ','.join(str(e) for e in sq)
            str2 += "]|["
            str2 += ','.join(str(e) for e in sq1)
            str2 += "]|["
            str2 += ','.join(str(e) for e in sq2)
            str2 += "]"
            newdata['Kmeans'] = sq
            newdata['Evolutionary'] = sq1
            newdata['Optimal'] = sq2

        pl.legend(loc="upper middle", ncol=clusters, borderaxespad=0., fontsize=10)
        pl.show()
        return str2


if __name__ == "__main__":
    result = ""
    tt = kmeansT()
    tt.test0()
    result += tt.test01()
    result += tt.test31()
    print(result)

    # sender = Sender()
    # sender.sendMessage(result)
    # tt.test31()
    # test.test01()

#     com=Compare(lastCenters,pred,currCenters,pred1)
#     print com.BestMatch()
#     com.match()
#     print com.pair
#     for i in range(3):
#         res[res['pred1']==i]['pred1']=com.pair[i]
#     print len(res[res['pred1']==res['pred']])
#     print len(res)
