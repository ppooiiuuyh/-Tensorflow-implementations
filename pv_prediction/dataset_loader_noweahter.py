import numpy as np
import pandas as pd


class Data:
    def __init__(self,cur_date,pv_chunk, rain_forecast, sky_forecast,pv_label,maxGen):
        self.cur_date = cur_date
        self.pv_chunk = pv_chunk
        self.rain_forecast = rain_forecast
        self.sky_forecast = sky_forecast
        self.pv_label = pv_label
        self.maxGen = maxGen


    def getScalarLabel(self,numClasses):
        interval = self.maxGen/numClasses
        scalarLabel = int(self.pv_label/interval)
        return scalarLabel

    def getOnehotLabel(self, numClasses):
        interval = self.maxGen / numClasses
        scalarLabel = int(self.pv_label / interval)

        return np.eye(numClasses)[scalarLabel]

    def getLinearShapeInput(self):
        output = [p for p in self.pv_chunk]
        output = np.array(output)
        output = output.reshape(output.shape[0]*output.shape[1]).tolist()

        #output.append(self.rain_forecast)
        #output.append(self.sky_forecast)
        output = np.array(output)
        output = output.reshape(len(output))
        return output

    def get2DShapeInput(self):
        output = [p for p in self.pv_chunk]
        output = np.array(output)
        output = output.reshape(output.shape[0],output.shape[1],1)
        return output

class Dataset_loader:
    def __init__(self,pvdir,raindir,skydir,trainset_ratio = 0.7,maxGen = 1490 , numClasses = 149):
        #do not use year,month,day
        self.cur_date = np.array(pd.read_csv(pvdir))[:,0:4].astype(int)
        self.pv = np.array(pd.read_csv(pvdir))[:,3:].astype(float)
        self.pv_copy = np.array(pd.read_csv(pvdir))[:,3:].astype(float)
        self.rain = np.array(pd.read_csv(raindir))[:,5:].reshape(-1,1).astype(float)
        self.sky = np.array(pd.read_csv(skydir))[:,5:].reshape(-1,1).astype(float)
        #self.sky = np.array(pd.read_csv(skydir))[:,3:].astype(float)

        self.duration = 6
        self.maxGen = maxGen
        self.numClasses = numClasses

        self.attribute_wise_normalization(self.pv,maxList=[24,1490,None,None,None,None])
        self.attribute_wise_normalization(self.rain)
        self.attribute_wise_normalization(self.sky)
        self.dataset = self.genDataset()


        self.trainset_ratio = trainset_ratio
        self.trainset =[]
        self.testset  =[]
        self.split_dataset()

        dist = [0 for d in range(10)]
        for d in self.dataset:
            dist[d.getScalarLabel(10)] +=1
        print(dist)

    def genDataset(self):
        dataset = []
        for i in range(self.duration,self.pv.shape[0]): #i = current +1
            cur_date = self.cur_date[i-1]
            pv_chunk = self.pv[i-self.duration: i]
            rain_forecast = self.rain[i-1]
            sky_forecast = self.sky[i-1]
            pv_label = self.pv_copy[i,1]

            if(pv_label != 0 and pv_label <1490):
                dataset.append(Data(cur_date,pv_chunk,rain_forecast,sky_forecast,pv_label,self.maxGen))
        return dataset

    def attribute_wise_normalization(self,arr,maxList=None, minList=None): #normalize 0 to max
        if(maxList != None and len(maxList) != arr.shape[1]): raise AssertionError("maxList,arr shape do not match")
        if(minList != None and len(minList) != arr.shape[1]): raise AssertionError("minList,arr shape do not match")
        if(maxList == None): maxList = [None for i in range(arr.shape[1])]
        if(minList == None): minList = [None for i in range(arr.shape[1])]

        for i in range(arr.shape[1]):
            if (maxList[i] == None): maxList[i] = np.amax(arr[:, i])
            if (minList[i] == None): minList[i] = np.amin(arr[:, i])
            arr[:,i] = arr[:,i] / ( maxList[i] - minList[i])


    def split_dataset2(self,seed = 10):
        np.random.seed(seed)

        p = np.random.permutation(len(self.dataset))
        dataset_rand = np.array(self.dataset)[p]

        for d in dataset_rand:
            if(np.random.random() < self.trainset_ratio):
                self.trainset.append(d)
            else:
                self.testset.append(d)

    def split_dataset(self):
        for e,d in enumerate(self.dataset):
            if(e<len(self.dataset)*self.trainset_ratio):
                self.trainset.append(d)
            else:
                self.testset.append(d)







def module_test():
    dl = Dataset_loader(
        pvdir = "./data/pv_2016_processed.csv",
        raindir= "./data/rain_2016_processed.csv",
        skydir="./data/sky_2016_processed.csv")
    print(dl.dataset[0].pv_chunk)
    print(dl.dataset[0].rain_forecast)
    print(dl.dataset[0].sky_forecast)
    print(dl.dataset[0].cur_date)
    print(dl.dataset[0].pv_label)

    print(len(dl.trainset))
    print(len(dl.testset))
    print(dl.dataset[0].getScalarLabel(149))
    pass

if __name__ == "__main__":
    module_test()