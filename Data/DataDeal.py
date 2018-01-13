import logging
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
import jieba
import os
import json
import pickle
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger = logging.getLogger("123")
import time


# ltp=remote_ltp_service.RemoteLtpService()
stopWord = []
with open("./stopWord_old", 'r') as f:
    for word in f.readlines():
        word = word.strip().replace("\n", "")
        stopWord.append(word)

def getToken(sentence,index):
    '''

    :param sentence: 
    :return: 
    '''
    words=jieba.cut(sentence)
    wordlist=words
    wordlist = [word for word in wordlist if word not in stopWord]
    return index,wordlist,sentence

class GetData(object):

    def __init__(self):
        self.DataDict={}
        self.index=0

    def _readJson(self,JsonFile):
        '''
        读取json文件
        :param JsonFile: 
        :return: 
        '''
        with open(JsonFile, 'r', encoding='utf-8') as f:

            for ele in f.readlines():
                js = json.loads(ele)
                # print("*"*20)
                for e in js["documents"]:
                    if e["is_selected"]:
                        self.DataDict[self.index] = {"segmented_title": e["segmented_title"],
                                           "segmented_paragraphs": e["segmented_paragraphs"]}
                        self.index += 1

    def get_origin_data(self,filedir):
        '''
        获取原始数据
        :return: 
        '''
        if not os.path.exists(filedir):
            _logger.info("文件不存在!")

        list = os.listdir(filedir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(filedir, list[i])
            if os.path.isfile(path):
                self._readJson(path)

        pickle.dump(self.DataDict,open("DataDict.p",'wb'))

    def loadData(self):
        self.DataDict=pickle.load(open("./DataDict.p",'rb'))

    def TrainWord2vec(self):

        datas=[]
        for k,v in self.DataDict.items():
            datas.append(self.dealStopWord(v["segmented_title"]))

        model_dm = Word2Vec(datas, min_count=1, window=5, size=100,workers=8)
        model_dm.train(datas, total_examples=model_dm.corpus_count, epochs=10)
        model_dm.save("./word2vec.model")

    def sent2vec(self):
        '''
        将句子转化为词向量累加
        :return: 
        '''
        w2v_model=Word2Vec.load("./word2vec.model")
        DataVecDict={}
        for k,v in self.DataDict.items():
            words=v["segmented_title"]
            words=self.dealStopWord(words)
            ss=np.zeros((100,))
            sum_nm=0
            for e in words:
                if e in w2v_model:
                    ss+=w2v_model[e]
                    sum_nm+=1
            if sum_nm==0:
                pass
            else:
                vec=ss/float(sum_nm)
                DataVecDict[k]=vec
        pickle.dump(DataVecDict,open("./DataVecDict.p",'wb'))



    def dealStopWord(self,wordlist):
        '''
        去停用词
        :param wordlist: 
        :return: 
        '''
        newWords=[]
        for word in wordlist:
            if word not in stopWord:
                newWords.append(word)
        return newWords




if __name__ == '__main__':
    # s_time=time.time()
    #
    # gt=GetData()
    # gt.loadData()
    # gt.sent2vec()
    # gt.TrainWord2vec()
    #
    # e_time=time.time()
    #
    #
    #
    # print("all time",e_time-s_time)
    DataVecDict=pickle.load(open("./DataVecDict.p",'rb'))
    for k,v in DataVecDict.items():
        print(k)
