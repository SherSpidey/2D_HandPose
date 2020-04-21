from old_ver.V1.data_funs import *
import numpy as np


class DS(object):
    def __init__(self, data_dir, batch_size, datasize, mode='training'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.datasize=datasize
        self.all_truns = int(datasize / batch_size)
        self.turn = 0
        self.mode = mode
        self.images = np.array([])
        self.annotations = np.array([])
        self.loadlabels()

    # load annotations
    def loadlabels(self):
        self.annotations = load_annotation(self.data_dir)

    # output adjustable batch_size's data
    def NextBatch(self):
        """

        :return:
        """
        """self.turn = np.random.randint(0, self.datasize-self.batch_size)
        self.images = load_image(self.data_dir, index=self.turn, batch_size=self.batch_size)
        self.turn += 1
        if self.turn == self.all_truns:
            self.turn = 1
        return self.images, self.annotations[self.turn:self.turn + self.batch_size, :]"""
        image=[]
        annotation=[]
        for _ in range(self.batch_size):
            self.turn = np.random.randint(0, self.datasize)
            mid_image=load_image(self.data_dir, index=self.turn, batch_size=1)
            image.append(mid_image[0])
            annotation.append(self.annotations[self.turn,:])

        image=np.array(image)
        annotation=np.array(annotation)

        return image,annotation

