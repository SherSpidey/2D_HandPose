from old_ver.V1.DS import DS
from old_ver.V1.SetValues import SV
from old_ver.V1.model_units_funs import *
from old_ver.V1.data_funs import *

M=DS(SV.dataset_main_path,SV.batch_size,32560)
print(M.annotations.shape)
"""for i in range(150):
    for j in range(100):
        a,b=M.NextBatch()
        #print(b.shape)
        c=generate_heatmap(224,28,b)
    print("epsoid %d over!"%(i+1))

a,b=M.NextBatch()
c=generate_heatmap(224,28,b)
d=get_coods_v2(c)
print(b)
print(d)
reshow(a,b,SV.batch_size)
reshow(a,d,SV.batch_size)"""
img,lab=M.NextBatch()
reshow(img,lab,SV.batch_size)
img=img[0,:,:,:]
img=cv2.resize(img,(368,368),interpolation=cv2.INTER_CUBIC)[np.newaxis,:,:,:]
#an=generate_heatmap(224,28,lab)
#lab=get_coods_v2(an)
lab=lab[0,:,:]
lab=lab*(368/224)
lab=lab[np.newaxis,:,:]
reshow(img,lab,SV.batch_size)