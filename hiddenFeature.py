import sys
import csv
import os
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans

def opencsv(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        # 读取一行，下面的reader中已经没有该行了
        head_row = next(reader)
        plus=0
        arr=[]
        arr2=[]
        for row in reader:
            # 行号从2开始
            plus+=int(float(row[1]))
            if(reader.line_num%24==1):
                arr2.append(plus)
                plus=0
    return arr2

def searchFiles(rootDir,x):
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            filename=os.path.join(root,file)
            x.append(opencsv(filename))
            for dir in dirs:
                searchFiles(dir)

def sparse_loss(h, leng):
    q = tf.reduce_mean(h, 1)
    p = tf.constant(0.05, shape=[1, leng])
    kl = p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))
    return tf.reduce_sum(kl)


class Hidden():
    def __init__(self,learning_rate,training_epoch,batch_size,n_hidden,y):
        self.learning_rate = learning_rate#0.001
        self.training_epoch = training_epoch#10000
        self.batch_size =batch_size# 100
        self.n_hidden =n_hidden
        self.n_input = y.shape[1]
        self.y=y


        self.X = tf.placeholder(tf.float32, [None, self.n_input])

        self.W_encode = tf.Variable(tf.random_normal([self.n_input, self.n_hidden]))
        self.b_encode = tf.Variable(tf.random_normal([self.n_hidden]))

        self.encoder = tf.nn.sigmoid(
            tf.add(tf.matmul(self.X, self.W_encode), self.b_encode))

        self.W_decode = tf.Variable(tf.random_normal([self.n_hidden, self.n_input]))
        self.b_decode = tf.Variable(tf.random_normal([self.n_input]))

        self.decoder = tf.nn.sigmoid(
            tf.add(tf.matmul(self.encoder, self.W_decode), self.b_decode))

        self.cost = tf.reduce_mean(tf.pow(self.X - self.decoder, 2)) + sparse_loss(self.encoder, y.shape[0]//3)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.total_cost = 0

    def fit(self):


        for epoch in range(self.training_epoch):

            hh = np.random.randint(0, self.y.shape[0], size=y.shape[0]//3)
            batch_x = self.y[hh]
            _, cost_val = self.sess.run([self.optimizer, self.cost],
                                   feed_dict={self.X: batch_x})
            self.total_cost += cost_val
            if epoch % 1000 == 1:
                print('Epoch:', '%04d' % (epoch + 1),
                      'Avg. cost =', '{:.4f}'.format(cost_val))

        print('Learning has finished!')
        encode, samples = self.sess.run([self.encoder, self.decoder],
                                   feed_dict={self.X: self.y})
        return encode,samples

    def reverse(self,encoderdata):
        samples = self.sess.run([ self.decoder],
                           feed_dict={self.encoder:encoderdata})
        return samples[0]


if __name__ =='__main__':
    rootDir = sys.argv[1]     #文件目录
    learning_rate=0.0005
    training_epoch=10000
    batch_size=100
    n_hidden=int(sys.argv[2])#特征维度
    cluser_nums=int(sys.argv[3])#聚类数


    x = []
    searchFiles(rootDir, x)
    x = np.array(x)
    print(x.shape)
    y = []
    for i in range(len(x)):
        tmp = (x[i] - np.min(x[i])) / (np.max(x[i]) - np.min(x[i]))
        y.append(tmp)
    y = np.array(y)
    print(y.shape)

    hidden1 = Hidden(learning_rate, training_epoch, batch_size, (n_hidden + y.shape[1]) // 2, y)  # learning_rate、迭代次数、单次迭代缓存大小、隐藏层维数
    encode1, samples1 = hidden1.fit()
    hidden2 = Hidden(learning_rate, training_epoch, batch_size, n_hidden, encode1)  # learning_rate、迭代次数、单次迭代缓存大小、隐藏层维数
    encode2, samples2 = hidden2.fit()#encode2为隐藏层特征



    y_pred = KMeans(n_clusters=cluser_nums).fit(encode2)  # 此处记录聚类数
    print(y_pred)#聚类的label

    print(y_pred.cluster_centers_.shape)#聚类的中心（隐藏层）
    reverse2 = hidden2.reverse(y_pred.cluster_centers_)
    reverse = hidden1.reverse(reverse2)#聚类的中心（还原后，但是不一定好看）
    print(reverse.shape)

    np.savetxt("HiddenFeatures.txt", encode2, fmt="%8f")
    np.savetxt("ClusteringResult.txt", y_pred.labels_, fmt="%d")