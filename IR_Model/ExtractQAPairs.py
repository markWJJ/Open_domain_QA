'''
从QA base中获取问句对应的QA对
'''
import numpy as np
import pickle
import jieba
from gensim.models.word2vec import Word2Vec

from IR_Model.basicClass import basicExtract
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger = logging.getLogger("123")
import time


# ltp=remote_ltp_service.RemoteLtpService()
stopWord = []
with open("../Data/stopWord_old", 'r') as f:
    for word in f.readlines():
        word = word.strip().replace("\n", "")
        stopWord.append(word)

class ExtractQA(basicExtract):

    def __init__(self):
        pass

    def loadModel(self):
        '''
        加载词向量
        :return: 
        '''
        self.W2v=Word2Vec.load("../Data/word2vec.model")
        self.DataDict=pickle.load(open("../Data/DataDict.p",'rb'))
        self.DataVecDict=pickle.load(open("../Data/DataVecDict.p",'rb'))

    def getTokenVec(self,sentence):

        sentences=jieba.cut(sentence)
        sents=[word for word in sentences if word not in stopWord]
        sentVec=np.zeros((100,))
        sumnum=0
        for word in sents:
            if word  in self.W2v:
                sentVec+=self.W2v[word]
                sumnum+=1
        if sumnum==0:
            _logger.info("sorry cant find")
        else:
            ss=sentVec/float(sumnum)
            return ss

    def computSim(self, sentvec):
        '''

        :param sentvec: 
        :return: 
        '''
        sentarray = np.array(sentvec)
        # print(sentarray.sape)
        result = []
        for k, v in self.DataVecDict.items():
            vec2 = np.array(v)

            num = float(np.dot(sentarray, vec2))  # 若为行向量则 A * B.T
            denom = np.linalg.norm(sentarray) * np.linalg.norm(vec2)
            cos = num / denom  # 余弦值
            sim = 0.3 + 0.7 * cos
            # dist = np.linalg.norm(sentarray - vec2)
            # sim= 1.0 / (1.0 + dist)
            result.append([k, sim, self.DataDict[k]["segmented_title"]])
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def extract(self,sentence,**kwargs):
        sentVec=self.getTokenVec(sentence)
        simResult=self.computSim(sentVec)
        return simResult



if __name__ == '__main__':
    extraQA=ExtractQA()
    extraQA.loadModel()
    while True:
        sent=input("输入")
        vec=extraQA.getTokenVec(sent)
        res=extraQA.computSim(vec)
        index=0
        for e in res:
            if index<20:
                print(e)
                index+=1

