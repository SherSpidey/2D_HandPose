import cv2
import json
import os
import numpy as np
import math


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


# load annotations from json file
def load_json(f):
    file_assert(f)
    with open(f, 'r') as J:
        print("Loading %s...." % f)
        return json.load(J)


# load dataset images
def load_data_image(datadir, index=0, num=1, mode='training'):
    images = []
    assert index < datasize(mode), "Out of data number!"
    # load several(batch_size) pictures from dataset
    image_root_path = os.path.join(datadir, mode, 'rgb')
    for i in range(index, index + num):
        image_path = os.path.join(image_root_path, "%08d.jpg" % i)
        file_assert(image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (368, 368), cv2.INTER_LANCZOS4)
        if num == 1:
            images = image
            break
        images.append(image)
    images = np.array(images)
    return images


# Transfrom 3D coordinates into 2D coordinates
def anno_trans(xyz, K):
    xyz = np.array(xyz)
    K = np.array(K)

    F = np.matmul(K, xyz.T).T
    z = F[:, -1:]
    xy = F[:, :2]
    return xy / z


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
    annotation = np.array(annotation) * (368 / 224)
    annotation = annotation.astype(int)

    return annotation


# draw skeleton of the hand
def draw_skeleton(img, coords):
    colors = [[102, 102, 102],
              [102, 0, 0],
              [153, 0, 0],
              [204, 0, 0],
              [255, 0, 0],
              [102, 102, 0],
              [153, 153, 0],
              [204, 204, 0],
              [255, 255, 0],
              [0, 102, 51],
              [0, 153, 76],
              [0, 204, 102],
              [0, 255, 127],
              [0, 51, 102],
              [0, 76, 153],
              [0, 102, 204],
              [0, 127, 255],
              [102, 0, 102],
              [153, 0, 153],
              [178, 0, 204],
              [255, 0, 255]]

    # color between two key points
    bones = [((0, 1), colors[1]),
             ((1, 2), colors[2]),
             ((2, 3), colors[3]),
             ((3, 4), colors[4]),

             ((0, 5), colors[5]),
             ((5, 6), colors[6]),
             ((6, 7), colors[7]),
             ((7, 8), colors[8]),

             ((0, 9), colors[9]),
             ((9, 10), colors[10]),
             ((10, 11), colors[11]),
             ((11, 12), colors[12]),

             ((0, 13), colors[13]),
             ((13, 14), colors[14]),
             ((14, 15), colors[15]),
             ((15, 16), colors[16]),

             ((0, 17), colors[17]),
             ((17, 18), colors[18]),
             ((18, 19), colors[19]),
             ((19, 20), colors[20])]

    # draw key points
    for i in range(len(colors)):
        cv2.circle(img, center=(coords[i, 0], coords[i, 1]), radius=2, color=colors[i], thickness=-1)

    # draw skeleton points
    for bone, color in bones:
        x1 = coords[bone[0], :][0]
        x2 = coords[bone[1], :][0]
        y1 = coords[bone[0], :][1]
        y2 = coords[bone[1], :][1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        limb = cv2.ellipse2Poly((int((x1 + x2) / 2), int((y1 + y2) / 2)), (int(length / 2+0.5), 3), int(deg), 0, 360, 1)
        cv2.fillConvexPoly(img, limb, color=color)

#show the result
def show_result(img,annotation,num=1):
    if num==1:
        draw_skeleton(img,annotation)
        cv2.imshow("Result",img)
        if cv2.waitKey(0) == 'q':
            cv2.destroyAllWindows()
    else:
        for i in range(num):
            draw_skeleton(img[i], annotation[i])
            cv2.imshow("Result"+str(i+1), img[i])
        if cv2.waitKey(0) == 'q':
            cv2.destroyAllWindows()

#heat map operations
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

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) /2.0/gaussian_variance /gaussian_variance)

def generate_heatmap(input_size,heatmap_size,batch_labels,model="cpm_sk",gaussian_variance=1):
    """
    generate heatmap from joints
    """
    scale=input_size//heatmap_size
    batch_heatmap=[]

    if len(batch_labels.shape)==2:
        batch_labels=batch_labels[np.newaxis,:,:]

    for eachpic in range(batch_labels.shape[0]):
        heatmap=[]
        reverse_hm=np.zeros(shape=(int(heatmap_size),int(heatmap_size)))
        for joints in range(batch_labels.shape[1]):
            j_hm=make_gaussian(heatmap_size,gaussian_variance,batch_labels[eachpic][joints]//scale)
            heatmap.append(j_hm)
            #reverse_hm-=j_hm
        if model=="cpm":
            heatmap.append(reverse_hm)
        batch_heatmap.append(heatmap)

    batch_heatmap=np.array(batch_heatmap)
    #need to trans-shape to adapt the model output: 28x28X21
    batch_heatmap=np.transpose(batch_heatmap,(0,2,3,1))
    return batch_heatmap

def get_coods(stage_heatmap,joints=21,box_size=368):
    annotation=np.zeros((21, 2))
    heatmap=stage_heatmap[0,:,:,0:joints].reshape(46,46,21)
    heatmap=cv2.resize(heatmap,(box_size,box_size))
    for joint_num in range(21):
        joint_coord = np.unravel_index(np.argmax(heatmap[:, :, joint_num]),
                                       (box_size, box_size))
        annotation[joint_num,:]=[joint_coord[1],joint_coord[0]]
    annotation = annotation.astype(int)
    return annotation



