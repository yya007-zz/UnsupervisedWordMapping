from __future__ import print_function
import os
import os.path as osp
import numpy as np

class DataLoaderDisk_bi(object):
    def __init__(self,dataset_name,do_shuffle=True):
        self.dataset = np.load(dataset_name) 
        self.loader_a = DataLoaderDisk_mo(self.dataset[:,0],do_shuffle)
        self.loader_b = DataLoaderDisk_mo(self.dataset[:,1],do_shuffle)
        self.num = min(self.loader_a.num,self.loader_b.num)
    def next_batch(self, batch_size=1):
        return self.loader_a.next_batch(batch_size),self.loader_b.next_batch(batch_size)

class DataLoaderDisk_mo(object):

    def __init__(self,dataset,do_shuffle=True):

        self.dataset = dataset 
        self.num = dataset.shape[0]

        # create self.order which is the order to generate batch
        self.perm = do_shuffle
        self.permutation()
        self._idx = 0

    def next_batch(self, batch_size=1):
        """
        Create the next batch
        Input:
            batch_size: int; the batch size to generate
        Return:
            images_batch: np4darray (batch_size, fine_size, fine_size, 3); the image batch
            seg_labels_batch: np3darray (batch_size, fine_size, fine_size); the seg label batch
            labels_batch: np1darray (batch_size,); the label batch
        """

        if self._idx+batch_size>=self.num:
            self._idx = 0
            self.permutation()

        if batch_size==1:
            data = self.dataset[self._idx]
            newshape=[1]
            newshape.extend(data.shape)
            data = data.reshape(newshape)
        else:
            data = self.dataset[self._idx:self._idx+batch_size]
        self._idx += batch_size
        return data

    def permutation(self):
        # permutation
        if self.perm:
            self.order = np.random.permutation(self.num) 
        else:
            self.order = np.arange(self.num)
        return
