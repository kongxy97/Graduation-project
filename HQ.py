# -*- coding: utf8 -*-
'''
Created on 2017年1月6日

@author: maitian13
'''
# from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import distance
import numpy as np
# from scipy.stats.stats import pearsonr
class HistoryQ(object):
    def __init__(self,lastCenters,currCenters,lastPred,currPred,size):
        self.lastCenters=lastCenters;
        self.currCenters=currCenters;
        self.lastPred=lastPred;
        self.currPred=currPred;
        self.size=size;
        #self.size=currCenters.size/currCenters.ndim;
    def calc(self):
        #distance=euclidean_distances(self.currCenters,self.lastCenters);
        #mins=distance.min(axis=0);
        #pairs=distance.argmin(axis=1)
        unq,_=np.unique(self.lastPred, return_inverse=True)
        unq_cnts1=np.bincount(_)
        unq,_=np.unique(self.currPred, return_inverse=True)
        unq_cnts2=np.bincount(_)
        matrix=[[0 for col in range(self.size)] for row in range(self.size)];
        for i in range(self.size):
            for j in range(self.size):
                matrix[i][j]=abs(unq_cnts1[i]-unq_cnts2[j]);
        p=np.argmin(matrix,axis=1);
        sum=0.0;
        count1=np.sum(unq_cnts1);
        count2=np.sum(unq_cnts2);
        for i in range(self.size):
            #tmp=(float(unq_cnts1[i])/count1+float(unq_cnts2[p[i]])/count2)/2;
            #tmp=abs(unq_cnts1[i]-unq_cnts2[p[i]]);
            sum+=distance.euclidean(self.lastCenters[i],self.currCenters[p[i]]);
            #sum+=distance.euclidean(self.currCenters[i],self.lastCenters[p[i]])
            #print unq_cnts1[i],"->",unq_cnts2[pairs[i]]
        #return mins.sum()
        return sum;
    def calc2(self):
        distance=cosine_distances(self.currCenters,self.lastCenters);
        mins=distance.min(axis=0);
        pairs=distance.argmin(axis=1)
        return mins.sum()
    def calc3(self):
        unq,_=np.unique(self.lastPred, return_inverse=True)
        unq_cnts1=np.bincount(_)
        unq,_=np.unique(self.currPred, return_inverse=True)
        unq_cnts2=np.bincount(_)
        matrix=[[0 for col in range(self.size)] for row in range(self.size)];
        for i in range(self.size):
            for j in range(self.size):
                matrix[i][j]=abs(unq_cnts1[i]-unq_cnts2[j]);
        p=np.argmin(matrix,axis=1);
        sum=0.0
        for i in range(self.size):
            sum+=abs(unq_cnts1[i]-unq_cnts2[p[i]])
        return sum;
    def calc4(self):
        matrix=[[0.0 for col in range(self.size)] for row in range(self.size)];
        for i in range(self.size):
            for j in range(self.size):
                #matrix[i][j]=pearsonr(self.currCenters[i],self.lastCenters[j]);
                matrix[i][j]=cosine_distances(self.currCenters[i].reshape(1, -1),self.lastCenters[j].reshape(1, -1));
        mins=np.min(matrix,axis=0)
        return mins.sum();