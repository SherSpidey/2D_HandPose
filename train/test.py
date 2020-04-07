import numpy as np
from DS import DS
from SetValues import SV
from model_units_funs import *
import data_funs

M=DS(SV.dataset_main_path,SV.batch_size,32560,1)
print(M.annotations.shape)
'''for i in range(150):
    for j in range(100):
        a,b=M.NextBatch()
        #print(b.shape)
        
        c=generate_heatmap(224,28,b)
    print("epsoid %d over!"%(i+1))'''
a,b=M.NextBatch()
data_funs.reshow(a,b,SV.batch_size)
heatmap=[]
variance = np.arange(3, 0, -1)
variance=np.sqrt(variance)
for i in range(3):
    heatmap.append(generate_heatmap(SV.input_size,
                                                                   SV.heatmap_size,b,variance[i]))
heatmap = np.array(heatmap)
heatmap = np.transpose(heatmap,(1,0,2,3,4))
c = generate_heatmap(224, 28, b,np.sqrt(2))
print(heatmap[:,1,:,:]==c)
