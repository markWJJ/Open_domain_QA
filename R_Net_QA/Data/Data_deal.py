import numpy as np
import pickle
import os
global PATH
PATH=os.path.split(os.path.realpath(__file__))[0]

class DataDeal(object):
    def __init__(self,train_path,dev_path,test_path,dim,batch_size,Q_len,A_len,flag):
        self.train_path=train_path
        self.dev_path=dev_path
        self.test_path=test_path
        self.dim=dim
        self.Q_len=Q_len
        self.A_len=A_len
        self.batch_size=batch_size
        if flag=="train_new":
            self.vocab=self.get_vocab()
            self.vocab_array=np.random.random((len(self.vocab),self.dim))
            pickle.dump(self.vocab,open(PATH+"/vocab.p",'wb'))
            pickle.dump(self.vocab_array,open(PATH+"/vocab_array.p",'wb'))
        elif flag=="test" or flag=="train":
            self.vocab=pickle.load(open(PATH+"/vocab.p",'rb'))
            self.vocab_array=pickle.load(open(PATH+"/vocab_array.p",'rb'))
        self.index=0
    def get_vocab(self):
        '''
        构造字典
        :return: 
        '''
        train_file=open(self.train_path,'r')
        test_file=open(self.dev_path,'r')
        dev_file=open(self.test_path,'r')
        vocab={"NONE":0}
        index=1
        for ele in train_file:
            ele1=ele.replace("\t\t"," ").replace("\n","").replace("	"," ")
            for w in ele1.split(" "):
                w=w.lower()
                if w not in vocab:
                    vocab[w]=index
                    index+=1
        for ele in test_file:
            ele1=ele.replace("	"," ").replace("\n","")
            for w in ele1.split(" "):
                w=w.lower()
                if w not in vocab:
                    vocab[w]=index
                    index+=1
        for ele in dev_file:
            ele1=ele.replace("	"," ").replace("\n","")
            for w in ele1.split(" "):
                w=w.lower()
                if w not in vocab:
                    vocab[w]=index
                    index+=1
        train_file.close()
        dev_file.close()
        test_file.close()
        return vocab

    def sent2vec(self,sent,max_len):
        '''
        根据vocab将句子转换为向量
        :param sent: 
        :return: 
        '''

        sent=str(sent).replace("\n","")
        sent_list=[]
        real_len=len(sent.split(" "))
        for word in sent.split(" "):
            word=word.lower()
            if word in self.vocab:
                sent_list.append(self.vocab[word])
            else:
                sent_list.append(0)
        if len(sent_list)>=max_len:
            new_sent_list=sent_list[0:max_len]
        else:
            new_sent_list=sent_list
            ss=[0]*(max_len-len(sent_list))
            new_sent_list.extend(ss)
        sent_vec=np.array(new_sent_list)
        return sent_vec,real_len

    def sent2array(self,sentence,len):
        '''
        根据句子的向量和voca_array转化为矩阵形式
        :param sent_vec: 
        :return: 
        '''
        sent_vec,real_len=self.sent2vec(sentence,len)

        sent_array=np.zeros(shape=(sent_vec.shape[0],self.dim))
        for i in range(sent_vec.shape[0]):
            sent_array[i]=self.vocab_array[sent_vec[i]]

        return sent_array,real_len

    def shuffle(self,Q,A,label):
        '''
        将矩阵X打乱
        :param x: 
        :return: 
        '''
        ss=list(range(Q.shape[0]))
        np.random.shuffle(ss)
        new_Q=np.zeros_like(Q)
        new_A=np.zeros_like(A)
        new_label=np.zeros_like(label)
        for i in range(Q.shape[0]):
            new_Q[i]=Q[ss[i]]
            new_A[i]=A[ss[i]]
            new_label[i]=label[ss[i]]

        return new_Q,new_A,new_label


    def nex_batch(self):
        '''
        获取训练机的下一个batch
        :return: 
        '''

        train_file=open(self.train_path,'r')
        Q_list=[]
        A_list=[]
        label_list=[]
        train_sentcens=train_file.readlines()
        file_size=len(train_sentcens)
        Q_len_list=[]
        A_len_list=[]
        for sentence in train_sentcens:
            sentences=sentence.split("\t\t")
            #sentences=sentence.split("	")
            Q_sentence=sentences[0]
            A_sentence=sentences[1]
            label=sentences[2]
            Q_array,Q_len=self.sent2array(Q_sentence,self.Q_len)
            A_array,A_len=self.sent2array(A_sentence,self.A_len)

            Q_list.append(list(Q_array))
            A_list.append(list(A_array))
            Q_len_list.append(Q_len)
            A_len_list.append(A_len)

            label_list.append(int(label))
        train_file.close()
        result_Q=np.array(Q_list)
        result_A=np.array(A_list)
        result_Q_len_list=np.array(Q_len_list)
        result_A_len_list=np.array(A_len_list)

        result_label=np.array(label_list)

        result_Q,result_A,result_label=self.shuffle(result_Q,result_A,result_label)


        num_iter=int(file_size/self.batch_size)
        if self.index<=num_iter:
            return_Q=result_Q[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_A=result_A[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_label=result_label[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_Q_len=result_Q_len_list[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_A_len=result_A_len_list[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
        else:
            self.index=0
            return_Q=result_Q[0:self.batch_size]
            return_A=result_A[0:self.batch_size]
            return_Q_len = result_Q_len_list[0:self.batch_size]
            return_A_len = result_A_len_list[0:self.batch_size]
            return_label=result_label[0:self.batch_size]
        return return_Q,return_A,return_label

    def nex_batch_(self):
        '''
        获取训练机的下一个batch
        :return: 
        '''

        train_file=open(self.train_path,'r')
        Q_list=[]
        A_list=[]
        label_list=[]
        train_sentcens=train_file.readlines()
        file_size=len(train_sentcens)
        Q_len_list=[]
        A_len_list=[]
        for sentence in train_sentcens:
            sentences=sentence.split("\t\t")
            #sentences=sentence.split("	")
            Q_sentence=sentences[0]
            A_sentence=sentences[1]
            label=sentences[2]
            Q_array,_=self.sent2array(Q_sentence,self.Q_len)
            A_array,_=self.sent2array(A_sentence,self.A_len)

            Q_list.append(list(Q_array))
            A_list.append(list(A_array))
            if len(str(Q_sentence).split(" "))>=self.Q_len:
                Q_len_list.append(self.Q_len)
            else:
                Q_len_list.append(len(str(Q_sentence).split(" ")))

            if len(str(A_sentence).split(" ")) >= self.A_len:
                A_len_list.append(self.A_len)
            else:
                A_len_list.append(len(str(A_sentence).split(" ")))

            label_list.append(int(label))
        train_file.close()
        result_Q=np.array(Q_list)
        result_A=np.array(A_list)
        result_Q_len_list=np.array(Q_len_list)
        result_A_len_list=np.array(A_len_list)

        result_label=np.array(label_list)

        result_Q,result_A,result_label=self.shuffle(result_Q,result_A,result_label)


        num_iter=int(file_size/self.batch_size)
        if self.index<=num_iter:
            return_Q=result_Q[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_A=result_A[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_label=result_label[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_Q_len=result_Q_len_list[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_A_len=result_A_len_list[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
        else:
            self.index=0
            return_Q=result_Q[0:self.batch_size]
            return_A=result_A[0:self.batch_size]
            return_Q_len = result_Q_len_list[0:self.batch_size]
            return_A_len = result_A_len_list[0:self.batch_size]
            return_label=result_label[0:self.batch_size]
        return return_Q,return_A,return_label,return_Q_len,return_A_len


    def get_dev(self):
        '''
        读取验证数据集
        :return: 
        '''
        dev_file = open(self.dev_path, 'r')
        Q_list = []
        A_list = []
        label_list = []
        train_sentcens = dev_file.readlines()
        for sentence in train_sentcens:
            sentences=sentence.split("	")
            Q_sentence=sentences[0]
            A_sentence=sentences[1]
            label=sentences[2]
            Q_array=self.sent2array(Q_sentence,self.Q_len)
            A_array=self.sent2array(A_sentence,self.A_len)

            Q_list.append(list(Q_array))
            A_list.append(list(A_array))
            label_list.append(int(label))
        dev_file.close()
        result_Q=np.array(Q_list)
        result_A=np.array(A_list)
        result_label=np.array(label_list)
        return result_Q,result_A,result_label

    def get_test(self):
        '''
        读取测试数据集
        :return: 
        '''
        test_file = open(self.test_path, 'r')
        Q_list = []
        A_list = []
        label_list = []
        train_sentcens = test_file.readlines()
        for sentence in train_sentcens:
            sentences=sentence.split("	")
            Q_sentence=sentences[0]
            A_sentence=sentences[1]
            label=sentences[2]
            Q_array,_=self.sent2array(Q_sentence,self.Q_len)
            A_array,_=self.sent2array(A_sentence,self.A_len)

            Q_list.append(list(Q_array))
            A_list.append(list(A_array))
            label_list.append(int(label))
        test_file.close()
        result_Q=np.array(Q_list)
        result_A=np.array(A_list)
        result_label=np.array(label_list)
        return result_Q,result_A,result_label

    def  get_Q_array(self,Q_sentence):
        '''
        根据输入问句构建Q矩阵
        :param Q_sentence: 
        :return: 
        '''
        Q_array,_=self.sent2array(Q_sentence,self.Q_len)
        return Q_array
    def get_A_array(self,A_sentence):
        '''
        根据输入的答案句子构建A矩阵
        :param A_sentence: 
        :return: 
        '''
        return self.sent2array(A_sentence,self.A_len)[0]

if __name__ == '__main__':
    dd = DataDeal(train_path="./WikiQA-train-small.txt", test_path="./WikiQA-test.txt",
                            dev_path="./WikiQA-dev.txt",
                            dim=100, batch_size=128 ,Q_len=10, A_len=50, flag="train")
    Q,A,label,Q_len,A_len=dd.nex_batch_()

    print(A_len[:10])