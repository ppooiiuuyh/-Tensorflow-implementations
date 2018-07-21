import pandas as pd
import numpy as np
import math

#===========================================
# 1. parsing
#===========================================
d  = pd.read_csv("./data/pv_2016_gs.csv")
d = np.array(d)[1:,[0,14,12,13]] #시간, 태양광유효전력, 수평일사량, 경사면일사량
d = np.array(d).astype(float)


d2 = np.array(d[0]).reshape(-1,d[0].shape[0])
for i in range(d.shape[0]):
    timeDif = int(d[i,0])%10000 - int(d2[-1,0])%10000
    if( timeDif == 15 or timeDif == 55 or timeDif ==-2345):
        d2 = np.append(d2,d[i].reshape(-1,d[i].shape[0]),axis=0)
print(d2)

'''# data per month '''
for m in range(13):
    count = 0
    for i in d2:
        if (math.floor(i[0] / 1000000) == 2016 * 100 + m):
            count += 1
    print(m, count)


#시간을 일 시 분 으로 쪼갬
d2 = np.append(d2[:,0].reshape(-1,1),d2, axis=1)
d2 = np.append(d2[:,0].reshape(-1,1),d2, axis=1)
d2[:,0] = np.floor(d2[:,0]%1000000/10000)#day
d2[:,1] = np.floor(d2[:,1]%10000/100)#hour
d2[:,2] = np.floor(d2[:,2]%100/1)#minute

print(d)

save = pd.DataFrame(d2,columns=["day","hour","minute","gen","H_irradiation","S_irradiation"])
save.to_csv("./data/pv_2016_gs_processed.csv",index=False)
