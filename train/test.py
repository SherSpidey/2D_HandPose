from operations import *
from data_model import DS
from config import SV

"""M=DS(SV.dataset_main_path,SV.batch_size)

img,ano=M.NextBatch()
#print(img/255.0-0.5)
hm=generate_heatmap(SV.input_size,SV.heatmap_size,ano)
co=get_coods(hm)
show_result([img,img],[ano,co],num=SV.batch_size+1)"""

#load_vedio("../test/hand.mp4")

