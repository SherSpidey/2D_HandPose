import numpy as np
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

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / (gaussian_variance ** 2))

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




