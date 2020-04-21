import cv2
import json
import os
import numpy as np

# basic init

def datasize(mode='training'):
    if mode == 'training':
        return 32560
    if mode == 'evaluation':
        return 3960

# file operation functions
def file_assert(f):
    text = 'File does not exists: %s' % f
    assert os.path.exists(f), text

def load_json(f):
    file_assert(f)
    with open(f, 'r') as J:
        print("Loading %s...." % f)
        return json.load(J)

# load images
"""def load_image(datadir, index=0, batch_size=10, mode='training'):
    image = []
    assert index < datasize(mode), "Out of data number!"

    # load several(batch_size) pictures from dataset
    image_root_path = os.path.join(datadir, mode, 'rgb')
    for i in range(index, index + batch_size):
        image_path = os.path.join(image_root_path, "%08d.jpg" % i)
        file_assert(image_path)
        image.append(cv2.imread(image_path))
    image = np.array(image)

    return image"""