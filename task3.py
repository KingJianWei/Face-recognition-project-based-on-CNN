import numpy as np
import cv2
import os,re

#获取整个文件图片路径
class ImTrans:
    def __init__(self,path='faceImageGray/wangjianwei/'):
        self.path = path
        
    def getimgnames(self):
        filenames = os.listdir(self.path)
        imgnames = []
        for i in filenames:
            if re.findall('^\d+\.jpg$',i)!=[]:
                imgnames.append(i)
        return imgnames   
    
    def getimgdata(self):
        imgnames = self.getimgnames()
        n = len(imgnames)
        data = np.zeros([n,32,32,3],dtype='float32')
        for i in range(n):
            img = cv2.imread(self.path+imgnames[i])
            da_new = cv2.resize(img,(32,32))
            da_new = da_new[:,:,:]/255 
            data[i,:,:,:] = da_new
        return data

#获取faceImageGray文件中的文件名
paths = os.listdir('faceImageGray/')
     
#将多个数据进行整合
Data = np.zeros([6000,32,32,3],dtype='float32')
Labels = np.zeros([6000],dtype='float32')
k = 0
for i in paths:
    imgtrans = ImTrans(path = 'faceImageGray/'+i+'/')
    data = imgtrans.getimgdata()
    Data[k*600:(k+1)*600,:,:,:] = data
    Labels[k*600:(k+1)*600] = k
#    for j in range(600):
#        Labels.append(i)
    k = k+1

#划分数据集
print('原始数据集数据的形状为：',Data.shape)
print('原始数据集标签的形状为：',len(Labels))
from sklearn.model_selection import train_test_split
data_train, data_test,labels_train, labels_test = train_test_split(Data, Labels, test_size=0.2, random_state=42)
print('训练集数据的形状为：',data_train.shape)
print('训练集标签的形状为：',len(labels_train))
print('测试集数据的形状为：',data_test.shape)
print('测试集标签的形状为：',len(labels_test))
np.save("data/data_train.npy", data_train)
np.save("data/labels_train.npy", labels_train)
np.save("data/data_test.npy", data_test)
np.save("data/labels_test.npy", labels_test)