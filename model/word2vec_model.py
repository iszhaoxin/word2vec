import numpy as np
import sys, gc
sys.path.append('../data')
import argparse, chainer
from chainer import reporter
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.links.loss import negative_sampling
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer.training import extensions
from optparse import OptionParser
import inspect
import dataIterator
import data_helper
import mylib.texthelper.myprint as mpr
from chainer.training import extensions
import mylib.texthelper.dataset as dataset

def convert(batch, device):
    center, context = batch
    return center, context


class SkipModel(chainer.link.Chain):
    def __init__(self, wordCounts, embed_size, lossfun=negative_sampling.NegativeSampling):
        # 这句话的定义是实现其父类chainer.link.Chain的init
        super(SkipModel,self).__init__()
        with self.init_scope():
            # self.embedings = np.random.uniform(0,1,(dataMes.vocabSize,embed_size)).astype(np.float32)
            self.embedings = L.EmbedID(
                len(wordCounts), embed_size, initialW=I.Uniform(1. / embed_size))
            # 负采样函数需要获取词频数据 self.wordCnt, 和词向量维度信息embed_size,这样才能构建一个第二层映射矩阵
            self.lossfun = negative_sampling.NegativeSampling(embed_size,wordCounts, sample_size=5)
    def __call__(self, xwords, pwords):
        self.mprint = mpr.MPrint("../log/model_logging.txt")
        shape = self.embedings(pwords).shape
        pwords_embed = self.embedings(pwords)
        # 负采样的__call__函数中获取的是 第一层中的映射,也就是词向量矩阵,第二层映射是在负采样类中进行的
        loss = self.lossfun(pwords_embed,xwords,reduce='sum')
        reporter.report({'loss': loss}, self)
        return loss

@profile
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    parser.add_argument('--val_rate', '-r', type=int, default=0.2,
                        help='Rate of valid set of all')
    parser.add_argument('--windows', '-w', type=int, default=2,
                        help='Window size to sample')
    parser.add_argument('--learning_rate', '-l', type=int, default=0.0005,
                        help='learning rate')
    args = parser.parse_args()

    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# windows: {}'.format(args.windows))
    print('# learning_rate: {}'.format(args.learning_rate))
    print('# val_rate: {}'.format(args.val_rate))

    print('')

    root = '../data/dataset/big/'
    dataMes = dataset.DataMes(fileName=root+'data')
    dataMes.filter(threshold=5)
    analysis = data_helper.DataAnalysis(dataMes)
    wordCounts = [analysis.wordDict[word] for word in analysis.wordList]
    train, val = analysis(val_rate=args.val_rate, window=args.windows)
    train_iter = dataIterator.dataIterator(analysis, train, batch_size=args.batchsize, repeat=True, shuffle=False)
    val_iter   = dataIterator.dataIterator(analysis, val, batch_size=args.batchsize, repeat=False, shuffle=False)
    # 配置优化以及训练的算法及参数
    vocab_size = train_iter.vocabSize
    model = SkipModel(wordCounts,args.unit)
    del analysis,train,val
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.learning_rate))
    updater = training.updater.StandardUpdater(train_iter, optimizer,converter=convert)
    trainer = training.Trainer(updater,(args.epoch, 'epoch'), out='./result/')

    # trainer.extend(extensions.Evaluator(val_iter, model, converter=convert))
    # trainer.extend(extensions.LogReport())
    # trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    # trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],'epoch', file_name='loss.png'))
    # trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='accuracy.png'))
    #
    # trainer.extend(extensions.ProgressBar())
    # trainer.run()
    # with open('./word2vec.model', 'w') as f:
    #     f.write('%d %d\n' % (len(train_iter.wordIndexInverse), args.unit))
    #     w = cuda.to_cpu(model.embedings.W.data)
    #     for i, wi in enumerate(w):
    #         v = ' '.join(map(str, wi))
    #         f.write('%s %s\n' % (train_iter.wordIndexInverse[i], v))


main()
