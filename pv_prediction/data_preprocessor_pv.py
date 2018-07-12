import pandas as pd
import numpy as np

#===========================================
# 1. parsing
#===========================================
d  = pd.read_csv("./data/pv_2016_.csv")
nd = np.array(d).astype(float)

'''
save = pd.DataFrame(nd)
save.to_csv("./data/pv_2016_processed.csv")
print(nd)'''