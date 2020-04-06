from data_funs import *


class DS(object):
    def __init__(self, data_dir, batch_size, datasize, mode='training'):
        self.data_dir = data_dir
        self.batch_size = batch_size
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
        self.images = load_image(self.data_dir, index=self.turn, batch_size=self.batch_size)
        self.turn += 1
        if self.turn == self.all_truns:
            self.turn = 1
        return self.images, self.annotations[(self.turn - 1)*self.batch_size:self.turn*self.batch_size, :]
