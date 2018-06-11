import numpy as np
import time, random
import mylib.texthelper.format as ft
import mylib.texthelper.decorator as decorator
import mylib.texthelper.dataset as dataset
import mylib.texthelper.myprint as mpr
import chainer, os
from chainer.iterators.serial_iterator import SerialIterator
from data_helper import *

class dataIterator(SerialIterator):
    # @profile
    def __init__(self, dataAnysis, dataset, batch_size, repeat=True, shuffle=False,val_rate=0.2):
        self.vocabSize = dataAnysis.vocabSize
        self.batch_size = batch_size
        # 保存这个是为了之后释放掉 self.dataAnysis的内存之后,还可以利用 self.wordIndex
        self.wordIndex = dataAnysis.wordIndex
        self.wordIndexInverse = dataAnysis.wordIndexInverse
        self.dataset = np.array(dataset)
        del dataset
        self.order = np.random.permutation(len(self.dataset)).astype(np.int32)
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self._repeat = repeat
        self._shuffle = shuffle
        gc.collect()
        del gc.garbage[:]
    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail
        # 获取本次从数据库中的截取范围
        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i:i_end]

        contexts = self.dataset[:,1].take(position,axis=0)
        # contexts = contexts.reshape(contexts.shape[0],1)
        center   = self.dataset[:,0].take(position,axis=0)

        if i_end >= len(self.order):
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
            if self._shuffle is True:
                np.random.shuffle(self.order)
        else:
            self.is_new_epoch = False
            self.current_position = i_end
        return np.array(center,dtype='int32'),np.array(contexts,dtype='int32')
    next = __next__
    @property
    def epoch_detail(self):
        # print self.epoch, self.current_position
        return self.epoch + self.current_position / len(self.dataset)
    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail
    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)


if __name__ == "__main__":
    @profile
    def main():
        root = './dataset/middle/'
        dataMes = dataset.DataMes(fileName=root+'data')
        dataMes.filter(threshold=5)
        analysis = DataAnalysis(dataMes)
        del dataMes
        train, val = analysis(val_rate=0.2, window=2)
        train_iterator = dataIterator(analysis, train, batch_size=1000, repeat=False, shuffle=False)
        val_iterator = dataIterator(analysis, val, batch_size=1000, repeat=False, shuffle=False)
        del train
        del val
    main()
