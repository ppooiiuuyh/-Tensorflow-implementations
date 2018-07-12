import pandas as pd
import numpy as np

#===========================================
# 1. parsing
#===========================================
d  = pd.read_csv("./data/rain_2016.csv")
nd = np.array(d)


deleteList =[]
for i in range(nd.shape[0]) :
    if len(nd[i][0]) > 3 :
        deleteList.append(i)
nd = np.delete(nd,deleteList,axis=0)

nd = np.insert(nd, 0, 2016, axis=1)
nd = np.insert(nd, 1, 0, axis=1)
nd = nd.astype(float)

print(nd)

month = 0
flag = 0
for i in range(nd.shape[0]) :
    if nd[i][2] == 1 and flag == 0 :
        flag = 1
        month += 1

    nd[i][1] = month

    if nd[i][2] == 2:
        flag = 0

for i in range(nd.shape[0]) :
    nd[i][3] -= 30
    nd[i][3] /= 100


deleteList =[]
for i in range(nd.shape[0]) :
    if(nd[i][4]) != 1:
        deleteList.append(i)

nd = np.delete(nd,deleteList,axis=0)


print(nd)

'''
save = pd.DataFrame(nd)
save.to_csv("./data/rain_2016_processed.csv")
'''
'''
for n in nd :
    print(n.astype(float))
'''