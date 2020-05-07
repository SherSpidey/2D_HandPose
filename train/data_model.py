from operations import *


class DS(object):
    def __init__(self, data_dir, batch_size, mode='training'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode
        self.turn = 0
        self.datasize = datasize(mode)
        self.annotations = np.array([])
        self.loadlabels()

    # load annotations
    def loadlabels(self):
        self.annotations = load_annotation(self.data_dir, mode=self.mode)

    # output adjustable batch_size's data
    def NextBatch(self):
        if self.mode == 'training':
            image = []
            annotation = []
            for _ in range(self.batch_size):
                turn = np.random.randint(0, self.datasize)
                mid_image = load_data_image(self.data_dir, index=turn)
                if self.batch_size == 1:
                    image = mid_image
                    annotation = self.annotations[turn, :]
                    break
                image.append(mid_image)
                annotation.append(self.annotations[turn, :])
            image = np.array(image)
            annotation = np.array(annotation)
            return image, annotation
        else:
            image = []
            annotation = []
            for _ in range(self.batch_size):
                mid_image = load_data_image(self.data_dir, index=self.turn)
                if self.batch_size == 1:
                    image = mid_image
                    annotation = self.annotations[self.turn, :]
                    break
                self.turn += 1
                image.append(mid_image)
                annotation.append(self.annotations[self.turn, :])
            image = np.array(image)
            annotation = np.array(annotation)
            return image, annotation
