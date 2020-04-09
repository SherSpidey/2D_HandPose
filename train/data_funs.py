import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
# import time
import json
import os


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
def load_image(datadir, index=0, batch_size=10, mode='training'):
    image = []
    assert index < datasize(mode), "Out of data number!"

    # load several(batch_size) pictures from dataset
    image_root_path = os.path.join(datadir, mode, 'rgb')
    for i in range(index, index + batch_size):
        image_path = os.path.join(image_root_path, "%08d.jpg" % i)
        file_assert(image_path)
        image.append(io.imread(image_path))
    image = np.array(image)

    return image


# load annotations

def load_annotation(data_dir, mode='training'):
    annotation = []

    # load camera annotations from json file
    xyz_path = os.path.join(data_dir, "%s_xyz.json" % mode)
    K_path = os.path.join(data_dir, "%s_K.json" % mode)
    xyz = load_json(xyz_path)
    K = load_json(K_path)

    # two file must match size
    assert len(xyz) == len(K), "Wrong annotation size !"

    for i in range(len(K)):
        annotation.append(anno_trans(xyz[i], K[i]))
    annotation = np.array(annotation)
    # print(annotation.shape)
    return annotation


# Operations on data

# Transfrom 3D coordinates into 2D coordinates
def anno_trans(xyz, K):
    xyz = np.array(xyz)
    K = np.array(K)

    F = np.matmul(K, xyz.T).T
    z = F[:, -1:]
    xy = F[:, :2]
    return xy / z


# Draw keypoint on the picture
def draw_keypoint(fig, coords, linewidth='1', flag='o'):
    # each keypoint's colors
    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])
    # color between two key points
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]
    # draw key points
    for i in range(len(colors)):
        fig.plot(coords[i, 0], coords[i, 1], flag, color=colors[i], )

    # draw connections between key points
    for bone, color in bones:
        kp1 = coords[bone[0], :]
        kp2 = coords[bone[1], :]
        fig.plot([kp1[0], kp2[0]], [kp1[1], kp2[1]], color=color, linewidth=linewidth)


# show the result picture
def reshow(pic, coords, num=1):
    if num==None:
        plt.imshow(pic)
        draw_keypoint(plt, coords)
        plt.show()
    for i in range(num):
        plt.imshow(pic[i])
        draw_keypoint(plt, coords[i])
        plt.axis('off')
        plt.show()
# a=load_annotation("../../../dataset/FreiHAND_pub_v2")
