class DataLoader(object):
    def __init__(self,data):
        self.data=data;
    def saveAsCSVTo(self,name):
        self.data.to_csv('../data/'+name,index=False);
    