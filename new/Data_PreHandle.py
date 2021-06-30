import datetime

import numpy as np
import pandas as pd

from ETL.Extract import DataExtracter
from ETL.Transform import Transformer

"""
              building_id  meter            timestamp  meter_reading
    0                   0      0  2016-01-01 00:00:00          0.000
    1                   1      0  2016-01-01 00:00:00          0.000
    2                   2      0  2016-01-01 00:00:00          0.000
    3                   3      0  2016-01-01 00:00:00          0.000
    4                   4      0  2016-01-01 00:00:00          0.000
    ...               ...    ...                  ...            ...
    20216095         1444      0  2016-12-31 23:00:00          8.750
    20216096         1445      0  2016-12-31 23:00:00          4.825
    20216097         1446      0  2016-12-31 23:00:00          0.000
    20216098         1447      0  2016-12-31 23:00:00        159.575
    20216099         1448      0  2016-12-31 23:00:00          2.850
    
    [20216100 rows x 4 columns]
"""


class DataHandle:
    # def __init__(self, data_source: str):
    # self.data = pd.read_csv(data_source)
    def __init__(self, data_set):
        self.data = data_set

    def getByTimeAndId(self, time: str, building_id: int):
        filter_data = self.data.loc[self.data['building_id'] == building_id]
        del filter_data['meter']
        del filter_data['building_id']
        filter_data = filter_data.groupby('timestamp')['meter_reading'].apply(sum).reset_index()
        print('------')
        print(filter_data)
        print(filter_data.loc[range(0, 2)])

        return filter_data

    def filterData(self):
        pass


if __name__ == '__main__':
    extractor = DataExtracter("../data/96data2016.csv")
    transformer = Transformer(extractor.data)
    date = '2015-01-01'
    t = str(datetime.datetime.strptime(date, "%Y-%m-%d").date())
    before = datetime.datetime.strptime(date, "%Y-%m-%d")
    lastData = transformer.filterTime(t)
    data_matrix = lastData.loc[:, '1':'93']
    deltaDays = 30
    for i in range(0, deltaDays):
        t1 = before + datetime.timedelta(days=i)
        print("------------------------", i, t1.strftime("%Y-%m-%d"))
        currData = transformer.filterTime(t1.strftime("%Y-%m-%d"))
        data_matrix1 = currData.loc[:, '1':'93']
        data_matrix1.to_csv('../data/daily/' + t1.strftime("%Y-%m-%d") + '.csv')


    # print("2016-01-01 10:00:00" < "2016-01-02 00:00:00")
    # data = [[0, 0, '2016-01-01 00:00:00', 1.2],
    #         [0, 0, '2016-01-01 01:00:00', 2.2],
    #         [0, 1, '2016-01-01 01:00:00', 2.1],
    #         [0, 1, '2016-01-01 02:00:00', 1.5],
    #         [0, 0, '2016-01-01 02:00:00', 1.1]]
    # df = pd.DataFrame(data, columns=['building_id', 'meter', 'timestamp', 'meter_reading'])
    # print(df)
    # date = "2016-01-01 00:00:00"
    # cur_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    # print(cur_date)
    # handle = DataHandle(df)
    # handle_data = handle.getByTimeAndId(cur_date.strftime("%Y-%m-%d %H:%M:%S"), 0)
    # print(handle_data)
    # cur_date = cur_date + datetime.timedelta(hours=1)
    # print(cur_date.strftime("%Y-%m-%d %H:%M:%S"))
