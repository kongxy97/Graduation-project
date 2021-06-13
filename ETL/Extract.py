# -*- coding: utf8 -*-
"""
Created on 2016年12月1日

@author: maitian13
"""
import pandas as pd


class DataExtracter(object):
    def __init__(self, path):
        self.path = path
        self.data = self.load()

    def load(self):
        try:
            return pd.read_csv(self.path, encoding='utf-8')
            # return self.data
        except Exception as e:
            print()

    def select(self, rang):
        return self.data.iloc[:, rang]

    def getColLen(self):
        return len(self.data.columns)


if __name__ == "__main__":
    extracter = DataExtracter("../data/96read201502.csv")
    rang = [0, 1, 2, 3] + range(4, extracter.getColLen(), 4)
    print(extracter.select(rang).axes)
