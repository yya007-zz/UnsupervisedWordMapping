from __future__ import print_function
import os
import os.path as osp
import numpy as np

class DataLoaderDisk_bi(object):
    def __init__(self, dataset_name, batch_size, do_shuffle=True):
        self.dataset = np.load(dataset_name)
        self.batch_size=batch_size
        self.loader_a = DataLoaderDisk_mo(self.dataset[:,0],batch_size,do_shuffle)
        self.loader_b = DataLoaderDisk_mo(self.dataset[:,1],batch_size,do_shuffle)
        self.num = min(self.loader_a.num,self.loader_b.num)
    def next_batch(self):
        return self.loader_a.next_batch(),self.loader_b.next_batch()

class DataLoaderDisk_mo(object):

    def __init__(self, dataset, batch_size, do_shuffle=True):

        self.dataset = dataset 
        self.num = dataset.shape[0]
        self.batch_size=batch_size

        # create self.order which is the order to generate batch
        self.perm = do_shuffle
        self.permutation()
        self._idx = 0

    def next_batch(self):
        """
        Create the next batch
        Input:
            batch_size: int; the batch size to generate
        Return:
            images_batch: np4darray (batch_size, fine_size, fine_size, 3); the image batch
            seg_labels_batch: np3darray (batch_size, fine_size, fine_size); the seg label batch
            labels_batch: np1darray (batch_size,); the label batch
        """

        if self._idx+self.batch_size>=self.num:
            self._idx = 0
            self.permutation()

        if self.batch_size==1:
            data = self.dataset[self._idx]
            newshape=[1]
            newshape.extend(data.shape)
            data = data.reshape(newshape)
        else:
            data = self.dataset[self._idx:self._idx+self.batch_size]
        self._idx += self.batch_size
        return data

    def permutation(self):
        # permutation
        if self.perm:
            self.order = np.random.permutation(self.num) 
        else:
            self.order = np.arange(self.num)
        return
