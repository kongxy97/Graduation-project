# coding=utf-8
import numpy as np
import pandas as pd

user_id = '1923270'


class LassoPre:
    def process(self, user_id):
        base_dir = "data/elecData/" + user_id + ".csv"
        # out_dir = "E:/pythonFile/20190317-1923270/1923270cwd"
        zero = pd.DataFrame(np.empty([1, 1]))

        for i in range(1, 97):
            df = pd.read_csv(base_dir, header=0)
            data = df[df.columns[i + 3]]  # 除了ycid和time，多了ycid,a,b 列所以
            data2 = data[1:].append(zero).reset_index(drop=True)  # 必须重新索引，不然连接时会多出一行
            data3 = data2[1:].append(zero).reset_index(drop=True)
            data4 = data3[1:].append(zero).reset_index(drop=True)
            data5 = data4[1:].append(zero).reset_index(drop=True)
            data6 = data5[1:].append(zero).reset_index(drop=True)
            data7 = data6[1:].append(zero).reset_index(drop=True)
            data8 = data7[1:].append(zero).reset_index(drop=True)
            out = pd.concat([data, data2, data3, data4, data5, data6, data7, data8], axis=1)
            out.columns = ['1', '2', '3', '4', '5', '6', '7', 'pre']
            rows = out.shape[0]
            out.drop([j for j in range(rows - 7, rows)], inplace=True)
            out_dir = "data/elecData/" + user_id + "cwd"
            out.to_csv(out_dir + str(i) + ".csv")
