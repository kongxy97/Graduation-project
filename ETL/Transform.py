# -*- coding: utf8 -*-
'''
Created on 2016年12月1日

@author: maitian13
'''
# import time
from ETL.Extract import DataExtracter 
from sqlite3 import datetime
from ETL.Loader import DataLoader
import pandas as pd
# from sklearn.cluster import KMeans
# import pandas as pd
# from MINNING.Evolutionary_Kmeans import KMeans
class Transformer(object):
    def __init__(self,data):
        self.data=data;
    def getColLen(self):
        return len(self.data.columns)
    def getMaxLoad(self):
        #self.data["maxload"]=self.data.iloc[:,range(4,24)].max(axis=1,skipna=False);
        length=self.getColLen()-1;
        self.data["maxload"]=self.data.ix[:,'1':'93'].max(axis=1,skipna=False);
        self.data["minload"]=self.data.ix[:,'1':'93'].min(axis=1,skipna=False);
        #self.data["mean"]=self.data.ix[:,'1':'93'].sum()/24;
        self.data["mean"]=self.data.ix[:,'sum']/24;
        self.data["var"]=self.data.ix[:,'1':'93'].var(axis=1,skipna=False);
        self.data['mid']=(self.data["maxload"]+self.data["minload"])/2-self.data["mean"];
        for i in self.data.ix[:,'1':'93'].columns:
            #self.data.loc[:,i]=self.data.loc[:,i].div(self.data.loc[:,"maxload"]);
            #self.data[i]=self.data[i].div(self.data["maxload"]);#这里有归一化...
            self.data[i].fillna(0.0,inplace=True)
        #self.data.fillna(0.0)
    def removeUseless(self):
        self.data=self.data.dropna(axis=0,how='any',inplace=False);
    def filterType1(self):
        self.data=self.data[self.data["b"]=="100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000"];
    def filterTime(self,exp):
        self.data.loc[:,"time"].astype('datetime64',copy=False);
        #self.data=self.data[self.data["time"]==exp&self.data["time"]];
        # tmpdata=self.data[self.data["time"]==exp];
        # pd.to_datetime(self.data.loc[:, "time"])
        tmpdata = self.data[self.data["time"] == exp];
        return tmpdata;
    def filterTimeBetween(self,exp1,exp2):
        self.data.loc[:,"time"].astype('datetime64',copy=False);
        self.data=self.data[self.data["time"]<=exp1];
        self.data=self.data[self.data["time"]>=exp2];
    def filterColums(self):
        self.data.drop(['a','b'], axis=1,inplace=True)
    def reset_index(self):
        self.data.reset_index(drop=True,inplace=True)
    def to_maxtrix(self):
        return self.data.ix[:,'1':'93'].as_matrix()
if __name__ == "__main__":
    extracter=DataExtracter("../data/xxxx.csv");
    rang=[0,1]+range(4,extracter.getColLen(),4);
    transformer=Transformer(extracter.select(rang));
    transformer.removeUseless()
    #transformer.filterType1()
    #transformer.filterColums()
    t=datetime.datetime(2016,1,1)
    t1=datetime.datetime(2015,1,1)
    transformer.filterTimeBetween(t.strftime("%Y-%m-%d"),t1.strftime("%Y-%m-%d"));
    #transformer.getMaxLoad();
    transformer.reset_index()
    loader=DataLoader(transformer.data);
    loader.saveAsCSVTo("96test2015.csv");