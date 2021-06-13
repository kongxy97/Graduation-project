# coding=utf-8
import pandas as pd


class LassoStatistics(object):
    def statistics(self, data):
        low = int(data['sum'].min()/100)*100
        upper = int(data['sum'].max()/100+1)*100
        scope=list(range(low,upper,100))
        group=pd.cut(data['sum'].values,scope,right=False)#分组区间,长度91
        freq=pd.DataFrame(group.value_counts(), columns=['freq'])
        freq['per'] = freq/freq['freq'].sum()
        freq['per'] = freq['per'].map(lambda x:"%.3f"%x)
        # print("lasso sta=", freq)
        return str(scope), str(freq['freq'].values.tolist()), str(freq['per'].values.tolist())
