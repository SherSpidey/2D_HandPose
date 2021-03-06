import numpy as np
import cv2
import tensorflow as tf


def make_gaussian(output_size, gaussian_variance=3, location=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, output_size, 1, float)
    y = x[:, np.newaxis]

    if location is None:
        x0 = y0 = output_size // 2
    else:
        x0 = location[0]
        y0 = location[1]

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) /2.0/gaussian_variance )

def generate_heatmap(input_size,heatmap_size,batch_labels,gaussian_variance=1):
    """
    generate heatmap from joints
    """
    scale=input_size//heatmap_size
    batch_heatmap=[]

    for eachpic in range(batch_labels.shape[0]):
        heatmap=[]
        for joints in range(batch_labels.shape[1]):
            heatmap.append(make_gaussian(heatmap_size,gaussian_variance,batch_labels[eachpic][joints]//scale))
        batch_heatmap.append(heatmap)

    batch_heatmap=np.array(batch_heatmap)
    #need to trans-shape to adapt the model output: 28x28X21
    batch_heatmap=np.transpose(batch_heatmap,(0,2,3,1))
    return batch_heatmap

def get_coords_from_heatmap(heatmap,scale=8):
    annotations=[]
    heatmap=np.transpose(heatmap,(0,3,1,2))
    for i in range(heatmap.shape[0]):
        joints=[]
        for j in range(heatmap.shape[1]):
            index=np.argmax(heatmap[i][j])
            x=index % heatmap[i][j].shape[1]
            y=index // heatmap[i][j].shape[1]
            if x<=3 or y <=3 or heatmap[i][j].shape[1]-x<=3 or heatmap[i][j].shape[1]-y<=3:
                heatmap[i][j][x][y]=0
                index = np.argmax(heatmap[i][j])
                x = index % heatmap[i][j].shape[1]
                y = index // heatmap[i][j].shape[1]
            joints.append([x*scale,y*scale])
        annotations.append(joints)
    annotations=np.array(annotations)
    return annotations

def get_coods_v2(stage_heatmap):
    annotation=np.zeros((21, 2))
    heatmap=stage_heatmap[0,:,:,:].reshape(28,28,21)
    heatmap=cv2.resize(heatmap,(368,368))
    for joint_num in range(21):
        joint_coord = np.unravel_index(np.argmax(heatmap[:, :, joint_num]),
                                       (368, 368))
        annotation[joint_num,:]=[joint_coord[1],joint_coord[0]]
    return annotation[np.newaxis,:]




