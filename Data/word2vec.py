
import gensim
from gensim.models.word2vec import Word2Vec
import multiprocessing
import numpy as np
import time
import pickle
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger = logging.getLogger("wrod2vec")


class NewsWord2Vec(object):

    def __init__(self):
        pass


    def _convert_data(self):
        '''
        标准化数据，转化为doc2vec能够处理的数据
        :return: 
        '''
        x_train=[]
        for i,ele in dict(self.Data).items():
            x_train.append(ele[0])
        return x_train

    def Train(self,startTime,endTime,title=False,content=False,min_count=1, window = 7, size = 200, sample=1e-3, negative=10,
              workers=4,epochs=20,savepath="word2vec.model"):
        '''
        训练doc2vec
        :param min_count: 
        :param window: 
        :param size: 
        :param sample: 
        :param negative: 
        :param workers: 
        :return: 
        '''
        _logger.info("*"*20+"begin train"+"*"*20)
        self.datadeal = DataDeal.GetJvlingData()
        # self.DataDict = self.datadeal.getTokenData(startTime,endTime,title=title)
        self.Data = self.datadeal.getTokenData(startTime,endTime,title=title)

        _logger.info("*"*20+"Data load finish"+"*"*20)
        self.TaggededDocument = gensim.models.doc2vec.TaggedDocument
        x_train=self._convert_data()
        _logger.info("Model parameter:min_count=%s, window = %s, size = %s, sample = %s, negative= %s,workers = %s,epochs = %s,savepath = %s"
                     %(min_count,window,size,sample,negative,workers,epochs,savepath))
        model_dm = Word2Vec(x_train, min_count=min_count, window=window, size=size, sample=sample, negative=negative,
                           workers=workers)
        model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=epochs)
        model_dm.save(savepath)
        pickle.dump(self.Data,open("model/DataDict.p",'wb'))
        # pickle.dump(self.Data,open("model/DataList.p",'wb'))

        return model_dm
    def VecIntegration(self,id,sentList):

        if len(sentList)==0:
            return [id,np.zeros((200,),dtype=np.float32)]
        else:
            sentvec=sum([self.model_dm[word] for word in sentList])/float(len(sentList))
            # sentvec=sum([self.model_dm[word] for word in sentList])

            return [id,sentvec]


    def LoadModel(self,savepath="word2vec.model"):
        self.model_dm = Word2Vec.load(savepath)
        self.DataDict = pickle.load(open("model/DataDict.p", 'rb'))
        self.newDataDict={}
        res=[]
        # pool=multiprocessing.Pool(processes=10)
        # for k,ele in self.DataDict.items():
        #     _logger.info("is process:%s"%k)
        #     res.append(pool.apply_async(self.VecIntegration,(k,ele[0])))
        # pool.close()
        # pool.join()
        # for da in res:
        #     ele=da.get()
        #     newDataDict[ele[0]]=ele[1]
        for k, ele in self.DataDict.items():
            index,sentvec=self.VecIntegration(k,ele[0])
            self.newDataDict[index]=sentvec


    def computSim(self,sentvec):
        '''
        
        :param sentvec: 
        :return: 
        '''
        sentarray=np.array(sentvec)
        print(sentarray.shape)
        result=[]
        for k,v in self.newDataDict.items():
            vec2 = np.array(v)

            num = float(np.dot(sentarray,vec2))  # 若为行向量则 A * B.T
            denom = np.linalg.norm(sentarray) * np.linalg.norm(vec2)
            cos = num / denom  # 余弦值
            sim=0.3+0.7*cos
            # dist = np.linalg.norm(sentarray - vec2)
            # sim= 1.0 / (1.0 + dist)
            result.append([k,sim,self.DataDict[k][1]])
        result.sort(key=lambda x:x[1],reverse=True)
        return result

    def Similar(self,sentence,threshold=0.5,topn=20,savepath="doc2vec.model"):
        '''
        计算sentence的相似句子
        :param sentence: 
        :return: 
        '''

        _,sentenceList,_=DataDeal.getToken(sentence,1)
        _logger.info("处理后文本：%s"%sentenceList)

        _,sentvec=self.VecIntegration(12,sentenceList)
        result=self.computSim(sentvec)

        for e in result[:20]:
            print(e)








if __name__ == '__main__':
    startTime = "2016-01-1"
    endTime = "2016-9-2"
    s_time=time.time()
    ndv=NewsWord2Vec()
    # ndv.Train(startTime=startTime,endTime=endTime,title=True,epochs=10,savepath='model/word2vec.model',window=3,
    #           workers=multiprocessing.cpu_count())
    #
    ndv.LoadModel(savepath='model/word2vec.model')


    # ndv.Similar("万科增持")
    # print(ndv.model_dm.most_similar("万科"))

    while True:
        sentence=input("输入")
        res=ndv.Similar(sentence)
        print(res)

        # index=input("输入index：")
        # resIndex=ndv.SimilarIndex(index)
        # print(resIndex)
    # for i in range(1000):
    #     print("\n")
    #     print(i)
    #     resIndex=ndv.SimilarIndex(i)
    #     print(resIndex)


    e_time=time.time()

    print("all time:%s"%(e_time-s_time))