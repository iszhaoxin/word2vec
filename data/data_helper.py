import numpy as np
import time, random
import mylib.texthelper.format as ft
import mylib.texthelper.decorator as decorator
import mylib.texthelper.dataset as dataset
import mylib.texthelper.myprint as mpr
import gc

class Evaluate():
    def __init__(self, fileName='questions-words.txt'):
        self.fileName = fileName
        self.questions = []
        with open(self.fileName,'r') as f:
            for line in f:
                self.questions.append(line.split())

    def evaluate(self):
        resultBool = [predict(QA[:-1])==QA[:-1] for QA in self.questions]
        return sum(resultBool)/len(resultBool)


def textStrip(text, strips):
    if strips is not None:
        for i in strips:
            text=text.strip(i)
    return text

def textStrip(text, strips):
    if strips is not None:
        for i in strips:
            text=text.strip(i)
    return text

class DataAnalysis():
    def __init__(self, dataMes):
        # 一些字符信息
        self.root = '/'.join(dataMes.fileName.split('/')[:-1])+'/'
        self.fileName = dataMes.fileName
        self.vocabSize = dataMes.vocabSize
        self.wordNum = dataMes.wordNum
        # 拥有大量数据的信息
        self.wordList = dataMes.wordList
        self.wordDict  = dataMes.wordDict
        self.wordIndex= dataMes.wordIndex
        self.wordIndexInverse = dataMes.wordIndex

        self.pairDict = 0
        self.wordINLine = []
        self.mprint = mpr.MPrint("../log/DataAnalysis_logging.txt")

    def __call__(self,val_rate, window, minicount=3, scoreThreshold=386.94):
        self.wordPair(minicount=minicount, scoreThreshold=scoreThreshold)
        return self.returnDataset(val_rate=val_rate, window=window)

    # @profile
    def wordPair(self, minicount, scoreThreshold):
        pairDict = {}
        scoreDict = {}
        # 消灭低频词,制作paircnt词典, 以及将wordInLine中的低频词换为,<unk>
        with open(self.fileName,'r') as f:
            for line in f:
                wordINLine = line.split()
                for i in range(len(wordINLine)-1):
                    i_exist = wordINLine[i] in self.wordDict
                    i_next_exist = wordINLine[i+1] in self.wordDict
                    if i_exist and i_next_exist:
                        wordPair = (wordINLine[i],wordINLine[i+1])
                        ft.addToDict(wordPair,pairDict)
                    else:
                        if i_exist is False:
                            wordINLine[i] = '<unk>'
                        if i_next_exist is False:

                            wordINLine[i+1] = '<unk>'
        # 根据scoreThreshold,筛选出保留的 pair,更新paircnt,以及创建 scoreDict
        for key in list(pairDict.keys()):
            w1,w2 = key
            score = self.wordNum*(pairDict[key]-minicount)/(self.wordDict[w1]*self.wordDict[w2])
            score = round(score,2)
            scoreDict.update({key:score})
            if score <= scoreThreshold:
                del pairDict[key]
        self.pairDict = pairDict
        # 将 pairDict 和 scoreDict 的信息写入文件
        self.mprint.pdict2f(self.pairDict,'self.pairDict')
        self.mprint.pdict2f(scoreDict,'self.scoreDict')

        # 将wordInLine中的pair词合并
        signal = True # True 意味着下一个正常append, False说明下一个是pair的一部分,跳过append
        unk_cnt = 0
        for i in range(len(wordINLine)-1):
            if wordINLine[i] == '<unk>':
                unk_cnt += 1
            if (wordINLine[i],wordINLine[i+1]) in self.pairDict:
                self.wordINLine.append('_'.join([wordINLine[i],wordINLine[i+1]]))
                signal = False
            else:
                if signal == True:
                    self.wordINLine.append(wordINLine[i])
                else:
                    signal = True
        # # 将 pair 的信息更新至 wordDict,wordIndex, wordIndexInverse, wordList, wordNum
        # # 这个里面没有考虑,在合并之后被完全组合掉的低频词,因为没有必要考虑
        while None in wordINLine:
            wordINLine.remove(None)
        # self.wordINLine = wordINLine
        self.updateMes()
        del wordINLine

        pair_cnt = sum(self.pairDict.values())
        self.mprint.pdict2f(self.wordDict, "self.wordDict")
        gc.collect()

    def updateMes(self):
        self.wordDict = {}
        self.wordList = []
        self.wordIndex = {}
        self.wordIndexInverse = {}
        for word in self.wordINLine:
            if word in self.wordDict:
                self.wordDict[word] += 1
            else:
                self.wordDict.update({word:1})
                self.wordList.append(word)
        for i in range(len(self.wordList)):
            self.wordIndex.update({self.wordList[i]:i})
            self.wordIndexInverse.update({i:self.wordList[i]})
        self.vocabSize   = len(self.wordList)
        self.wordNum = len(self.wordINLine)

    def wordProb(self):
        wordProb = {}
        for word in self.wordINLine:
            z_w = self.wordDict[word]/len(self.wordINLine)
            prob = np.sqrt(z_w*1000+1)*0.001/z_w
            wordProb.update({word:prob})
        self.mprint.pdict2f(wordProb, "wordProb")
        return wordProb

    def returnDataset(self, val_rate, window):
        val = 0
        train_dataset = []
        val_dataset = []

        wordProb = self.wordProb()
        for i in range(len(self.wordINLine)-window):
            thisWord = self.wordINLine[i]
            if thisWord != '<unk>' and random.random() <= wordProb[thisWord]:
                j=1
                while j<=window:
                    contextWord = self.wordINLine[i+j]
                    if contextWord != '<unk>':
                        if val%10>=2:
                            train_dataset.append((self.wordIndex[thisWord],\
                                self.wordIndex[contextWord]))
                            train_dataset.append((self.wordIndex[contextWord],\
                                self.wordIndex[thisWord]))
                        else:
                            val_dataset.append((self.wordIndex[thisWord],\
                                self.wordIndex[contextWord]))
                            val_dataset.append((self.wordIndex[contextWord],\
                                self.wordIndex[thisWord]))
                        val += 1
                    j += 1
        gc.collect()
        return train_dataset, val_dataset

if __name__ == '__main__':
    def func():
        root = './dataset/middle/'
        dataMes = dataset.DataMes(fileName=root+'data')
        dataMes.filter(threshold=5)
        dataMes.wordRecord(root+'words.txt')
        dataMes.wordMesRecord(root+'words_Mes.txt')
        analysis = DataAnalysis(dataMes)
        train, val = analysis(val_rate=0.2, window=2)
        print(len(train), len(val))

    func()
