import tensorflow as tf
import numpy as np

from R_Net_QA.Data import Data_deal
class CA(object):
    '''
    compare_aggregate模型
    '''
    def __init__(self,init_dim,hidden_dim,Q_len,A_len):
        self.init_dim=init_dim
        self.Q_len=Q_len
        self.A_len=A_len
        # 论文中的L维
        self.hidden_dim=hidden_dim
        self.Q=tf.placeholder(dtype=tf.float32,shape=(None,self.Q_len,self.init_dim))
        self.A=tf.placeholder(dtype=tf.float32,shape=(None,self.A_len,self.init_dim))
        self.label=tf.placeholder(dtype=tf.int32,shape=(None,))
        #self.Q_ = self.process(flag="N", x=self.Q, type="Q")
        #self.A_ = self.process(flag="N", x=self.A, type="A")
        self.Q_=self.Q
        self.A_=self.A
        #注意力矩阵
        self.H=self.attention(self.Q_,self.A_)
        # 合并A_和H
        self.T=self.compare(self.A_,self.H)

        #经lstm进行整合
        self.out=self.Lstm(self.T)

        #经softmax进行分类，分为两类：0 不匹配 1匹配
        softmax_w=tf.Variable(tf.random_normal((self.hidden_dim,2),dtype=tf.float32))
        softmax_b=tf.Variable(tf.random_normal((2,),dtype=tf.float32))

        logit=tf.add(tf.matmul(self.out,softmax_w),softmax_b)

        self.softmax_out=tf.nn.softmax(logit)
        self.loss=tf.losses.sparse_softmax_cross_entropy(self.label,self.softmax_out)

        self.train_op=tf.train.AdadeltaOptimizer(0.01).minimize(self.loss)


    def process(self,flag,x,type):
        '''
        preocess layer
        :return: 
        '''
        if flag=="LSTM":
            if type=="Q":
                self.lstm_input = tf.unstack(x, self.Q, 1)
            elif type=="A":
                self.lstm_input=tf.unstack(x,self.A,1)

            cell = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)
            outputs, states = tf.contrib.rnn.static_rnn(cell, self.lstm_input, dtype=tf.float32)
            outputs=tf.concat(outputs,1)
            return outputs
        else:
            w_i=tf.Variable(tf.random_normal(shape=(self.init_dim,self.hidden_dim)))
            w_u=tf.Variable(tf.random_normal(shape=(self.init_dim,self.hidden_dim)))
            b_i=tf.Variable(tf.random_normal(shape=(self.hidden_dim,)))
            b_u=tf.Variable(tf.random_normal(shape=(self.hidden_dim,)))

            outputs=tf.sigmoid(tf.add(tf.einsum("ijk,kl->ijl",x,w_i),b_i))*tf.tanh(tf.add(tf.einsum("ijk,kl->ijl",x,w_u),b_u))
            return outputs

    def attention(self,Q_,A_):
        '''
        标准softmax注意力机制模块
        :param Q_: 
        :param A_: 
        :return: 
        '''
        w_g = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
        b_g = tf.Variable(tf.random_normal(shape=(self.hidden_dim,)))
        logit=tf.add(tf.einsum("ijk,kl->ijl",Q_,w_g),b_g)#(None,Q,l)
        A_T=tf.transpose(A_,perm=[0,2,1])#(None,l,A)
        logit_=tf.einsum("ijk,ikl->ijl",logit,A_T)
        G=tf.nn.softmax(logit_)#(None,Q,A)
        G_T=tf.transpose(G,perm=[0,2,1])
        H=tf.einsum("ikj,ijl->ikl",G_T,Q_)#(None,A,l)
        return H

    def compare(self,A_,H):
        '''
        合并A_和注意力矩阵H,采用论文中的S UB M ULT +NN方式合并
        :param A_: 
        :param H: 
        :return: 
        '''
        flag="submult_nn"
        if flag=="submult_nn":
            w = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim, self.hidden_dim)))
            b = tf.Variable(tf.random_normal(shape=(self.hidden_dim,)))
            sub=(A_-H)*(A_-H)
            mult=A_*H
            con=tf.concat((sub,mult),2)#(None,A,2l)
            logit_1=tf.einsum("ijk,kl->ijl",con,w)
            logit=tf.add(logit_1,b)
            T=tf.nn.relu(logit)
            return T
        elif flag=="concat":
            T=tf.concat((A_,H),2)
            return T

    def Lstm(self,T):
        '''
        lstm层，将T组合为句向量进行分类
        :param T: 
        :return: 
        '''
        lstm_input = tf.unstack(T, self.A_len, 1)
        cell=tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                     initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                     state_is_tuple=False)
        outputs,states=tf.contrib.rnn.static_rnn(cell,lstm_input,dtype=tf.float32)
        return outputs[-1]

    def train(self,dd):
        def accuracy(predictions, labels):
            predict=np.argmax(predictions, 1)
            length=sum(1 for i,j in  zip(predict.flatten(),labels.flatten()) if int(i)==1 or int(j)==1)
            ss=sum([ 1 for i,j in zip(predict.flatten(),labels.flatten()) if int(i)==int(j) and int(i)!=0])

            #return 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]
            if float(length)==0.0:
                return 0.0
            else:
                return 100.0*(float(ss)/float(length))

        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False)
        #dev_Q,dev_A,dev_label=dd.get_dev()
        #test_Q,test_A,test_label=dd.get_test()
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            #saver.restore(sess,"./model.ckpt")
            sess.run(tf.global_variables_initializer())
            ini_acc=0.0
            for i in range(10000):
                Q,A,label=dd.nex_batch()
                train_softmax_out,train_loss,_=sess.run([self.softmax_out,self.loss,self.train_op],feed_dict={self.Q:Q,
                                                                                                              self.A:A,
                                                                                                              self.label:label})

                '''
                dev_softmax_out, dev_loss = sess.run([self.softmax_out, self.loss],
                                                 feed_dict={self.Q: dev_Q, self.A: dev_A, self.label: dev_label})
                test_softmax_out, test_loss  = sess.run([self.softmax_out, self.loss],
                                                 feed_dict={self.Q: test_Q, self.A: test_A, self.label: test_label})
                test_acc=accuracy(test_softmax_out,test_label)
                                    dev_acc=accuracy(dev_softmax_out,dev_label)
                                    print("验证误差%s ,验证准确率%s"%(dev_loss,dev_acc))
                print("测试误差%s ,测试准确率%s"%(test_loss,test_acc))
                '''

                train_acc = accuracy(train_softmax_out, label)
                print("第%s次迭代"%i)
                if train_acc>ini_acc:
                    ini_acc=train_acc
                    saver.save(sess,"./model.ckpt")

                    print("训练误差%s ,训练准确率%s"%(train_loss,train_acc))

                    print("\n")

    def test(self,Q,A):

        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess=sess,save_path="./model.ckpt")
            softmax_out= sess.run([self.softmax_out],feed_dict={self.Q: Q, self.A: A})
            print (softmax_out)

if __name__ == '__main__':

    init_dim=100
    batch_size=128
    Q_len=10
    A_len=50
    hidden_dim=100
    dd=Data_deal.DataDeal(train_path="../Data/output.txt",test_path="../Data/WikiQA-test.txt",dev_path="../Data/WikiQA-dev.txt",
                          dim=init_dim,batch_size=batch_size,Q_len=Q_len,A_len=A_len,flag="train_new")
    ca=CA(init_dim=init_dim,hidden_dim=hidden_dim,Q_len=Q_len,A_len=A_len)
    ca.train(dd)


