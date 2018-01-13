
from IR_Model.basicClass import basicExtract
import jieba
import pickle
from jieba.analyse.tfidf import TFIDF
import jieba.posseg as posss

class BM25(basicExtract):

    def __init__(self):
        self.DataDict=pickle.load(open("../Data/DataDict.p",'rb'))

        docLenAll=0
        docindex=0
        for k, v in self.DataDict.items():
            docindex+=1
            docLenAll+=len(v["segmented_title"])

        self.avgdl=float(docLenAll)/float(docindex)


    def extract(self,sentence,**kwargs):

        result=[]
        for k,v in self.DataDict.items():
            d_sent=v["segmented_title"]
            socre=self.socre(sentence,d_sent,self.avgdl)
            result.append([socre,d_sent])

        result.sort(key=lambda x:x[0],reverse=True)
        return result


    def R_silmilar(self,dl,avgdl,qf,f):
        '''
        计算词语w 与文档的相似度
        :param w: 
        :param d_sent: 
        :return: 
        '''
        k1=2.0
        k2=1.0
        b=0.75
        K=k1*(1.0-b+b*(dl/avgdl))
        r1=(f*(k1+1.0))/(f+K)
        r2=(qf*(k2+1.0))/(qf+k2)
        R=r1*r2

        return R

    def socre(self,query,d_sent,avgdl):
        '''
        计算问句 query 与 数据集 相应问答对的相似度
        :param sentence: 
        :param d_sent: 
        :return: 
        '''
        # TFIDF.extract_tags(query)
        query_=posss.cut(query.replace("\n",""))
        d_sent="".join(d_sent)
        d_sent=jieba.cut(d_sent.replace("\n",""))
        querys=[]
        posags=[]
        for w in query_:
            querys.append(w.word)
            posags.append(w.flag)
        d_sent=[w for w in d_sent]
        queryDict={}
        for word in querys:
            if word not in queryDict:
                queryDict[word]=1
            else:
                s=queryDict[word]
                s+=1
                queryDict[word]=s
        sentDict={}
        for word_ in d_sent:
            if word_ not in sentDict:
                sentDict[word_]=1
            else:
                s=sentDict[word_]
                s+=1
                sentDict[word_]=s

        dl=len(d_sent)




        wordSim=0.0
        for word,poss in zip(querys,posags):
            pf=queryDict[word]
            if word not in sentDict:
                f=0.0
            else:
                f=sentDict[word]
            sim=self.R_silmilar(float(dl),float(avgdl),float(pf),float(f))
            if poss not in ["r",'uj','y']:
                wordSim+=sim
        socre=wordSim/float(len(querys))
        return socre

if __name__ == '__main__':
    bm25=BM25()
    sent="京东和淘宝哪个好"
    # res=bm25.extract(sent)
    # index = 0
    # for e in res:
    #     if index<10:
    #         print(e)
    #         index+=1
    while True:
        sent=input("输入")
        res=bm25.extract(sent)
        index=0
        for e in res:
            if index<10:
                print(e)
                index+=1



