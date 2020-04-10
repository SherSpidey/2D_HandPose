from DS import DS
from SetValues import SV
from model_units_funs import *
from data_funs import *

M=DS(SV.dataset_main_path,SV.batch_size,32560,1)
print(M.annotations.shape)
"""for i in range(150):
    for j in range(100):
        a,b=M.NextBatch()
        #print(b.shape)
        c=generate_heatmap(224,28,b)
    print("epsoid %d over!"%(i+1))
"""
a,b=M.NextBatch()
c=generate_heatmap(224,28,b)
d=get_coords_from_heatmap(c)
reshow(a,b,SV.batch_size)
print(d.shape)
reshow(a,d,SV.batch_size)