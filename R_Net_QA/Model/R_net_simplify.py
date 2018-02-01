import numpy as np
import tensorflow as tf
import sys
#sys.path.append("../Data_1/Data_deal_1")
from Data_1 import Data_deal


class R_Net_simplify(object):
    '''
    R_net模型简化版
    '''
    def __init__(self,init_dim,hidden_dim,Q_len,A_len):
        self.init_dim=init_dim
        self.Q_len=Q_len
        self.A_len=A_len
        self.num_class=5
        self.hidden_dim=hidden_dim
        self.Q=tf.placeholder(dtype=tf.float32,shape=(None,self.Q_len,self.init_dim))
        self.A=tf.placeholder(dtype=tf.float32,shape=(None,self.A_len,self.init_dim))
        self.Q_seq_vec=tf.placeholder(dtype=tf.int32,shape=(None,))
        self.A_seq_vec=tf.placeholder(dtype=tf.int32,shape=(None,))
        self.label=tf.placeholder(dtype=tf.int32,shape=(None,A_len))
        self.Q_ = self.process(input_=self.Q,seq_len=self.Q_len,seq_vec=self.Q_seq_vec,flag="Q")
        self.A_ = self.process(input_=self.A,seq_len=self.A_len,seq_vec=self.A_seq_vec,flag="A")
        #注意力矩阵
        self.H=self.attention(self.Q_,self.A_)
        # 合并A_和H
        self.T=self.compare(self.A_,self.H)

        #经lstm进行整合
        self.out=self.Bi_Lstm(self.T,self.A_len,self.A_seq_vec)

        #经softmax进行分类，分为两类：0 不匹配 1匹配
        softmax_w=tf.Variable(tf.random_normal((2*self.hidden_dim,self.num_class),dtype=tf.float32))
        softmax_b=tf.Variable(tf.random_normal((self.num_class,),dtype=tf.float32))

        logit=tf.einsum("ijk,kl->ijl",self.out,softmax_w)
        logit=tf.add(logit,softmax_b)

        self.softmax_out=tf.nn.softmax(logit)
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            logit, self.label, self.A_seq_vec)
        self.logit = logit
        self.trans_params = trans_params  # need to evaluate it for decoding
        self.loss_op = tf.reduce_mean(-log_likelihood)
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.4).minimize(self.loss_op)

        '''
                label=tf.one_hot(self.label,self.num_class,1,0,2)
        label=tf.cast(label,tf.float32)
        label=tf.reshape(label,[-1])
        softmax_out=tf.reshape(self.softmax_out,[-1])
        self.loss=tf.losses.softmax_cross_entropy(label,softmax_out)

        self.train_op=tf.train.AdadeltaOptimizer(0.7).minimize(self.loss)
        '''



    def process(self,input_,seq_len,seq_vec,flag):
        '''
        preocess layer
        :return: 
        '''

        with tf.variable_scope(flag,reuse=False):
            lstm_input=tf.transpose(input_,[1,0,2])
            lstm_input = tf.unstack(lstm_input, seq_len, 0)
            cell_f = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)

            cell_b = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)
            #static_bidirectional_rnn
            (out,_,_)=tf.contrib.rnn.static_bidirectional_rnn(cell_f,cell_b,lstm_input,dtype=tf.float32,
                                                               sequence_length=seq_vec)

            out=tf.stack(out,0)
            out=tf.transpose(out,[1,0,2])
            return out

    def attention(self,Q_,A_):
        '''
        标准softmax注意力机制模块
        :param Q_: 
        :param A_: 
        :return: 
        '''

        w_g = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim, 2*self.hidden_dim)))
        b_g = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,)))
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
        flag="concat"
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

    def Bi_Lstm(self,T,seq_len,seq_vec):
        '''
        lstm层，将T组合为句向量进行分类
        :param T: 
        :return: 
        '''
        with tf.variable_scope("out_lstm",reuse=False):
            lstm_input=tf.transpose(T,[1,0,2])
            lstm_input = tf.unstack(lstm_input, seq_len, 0)
            cell_f = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)

            cell_b = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)
            #static_bidirectional_rnn
            (out,_,_)=tf.contrib.rnn.static_bidirectional_rnn(cell_f,cell_b,lstm_input,dtype=tf.float32,
                                                               sequence_length=seq_vec)

            out=tf.stack(out,0)
            out=tf.transpose(out,[1,0,2])
            return out
    def Verbit(self,logits,batch_size,trans_params,sequence_lengths):

        viterbi_sequences = []
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit_ = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit_, trans_params)
            viterbi_sequences += [viterbi_seq]
        viterbi_sequences = viterbi_sequences
        return viterbi_sequences

    def crf_acc(self, pre_label, real_label, rel_len):
        """

        :param best_path: 
        :param path: 
        :return: 
        """
        real_labels_all = []
        for label, r_len in zip(real_label, rel_len):
            real_labels_all.extend(label[:r_len])

        verbit_seq_all = []
        for seq, r_len in zip(pre_label, rel_len):
            verbit_seq_all.extend(seq[:r_len])

        best_path = verbit_seq_all
        path = real_labels_all
        # ss = sum([1 for i, j in zip(best_path, path) if int(i) == int(j)])
        # length = sum([1 for i in path if int(i) != 0])
        if len(best_path) != len(path):
            print("error")
        else:
            ss = sum([1 for i, j in zip(best_path, path) if int(i) == int(j) and int(i) != 0])
            length = sum([1 for i, j in zip(best_path, path) if int(i) != 0 or int(j) != 0])
            if length != 0:
                acc = (float(ss) / float(length))
            else:

                acc=0.00
            return acc


    def train(self,dd):
        def accuracy(predictions, labels):
            predict=np.argmax(predictions, 2)
            print(predict.shape)
            length=sum(1 for i,j in  zip(predict.flatten(),labels.flatten()) if int(i)!=0 or int(j)!=0)
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
            saver.restore(sess,"./model.ckpt")
            #sess.run(tf.global_variables_initializer())
            ini_acc=0.0
            for i in range(10000):
                Q,A,label,Q_len,A_len=dd.nex_batch_()
                Q_seq_vec=np.array(Q_len)
                A_seq_vec=np.array(A_len)
                train_logit,trans_params,train_loss,_=sess.run([self.logit,self.trans_params,self.loss_op,self.optimizer],feed_dict={self.Q:Q,
                                                                                                              self.A:A,
                                                                                                              self.Q_seq_vec:Q_seq_vec,
                                                                                                              self.A_seq_vec:A_seq_vec,
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
                verbit_seq = self.Verbit(logits=train_logit, batch_size=Q.shape[0], trans_params=trans_params,
                                         sequence_lengths=A_seq_vec)

                train_acc = self.crf_acc(verbit_seq, label,A_seq_vec)
                print("第%s次迭代"%i)
                print("训练误差%s ,训练准确率%s" % (train_loss, train_acc))

                if train_acc>ini_acc:
                    ini_acc=train_acc
                    print("save model")
                    saver.save(sess,"./model.ckpt")
                    print("\n")

    def test(self,Q,A,Q_len,A_len):

        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess=sess,save_path="./model.ckpt")
            logit,trans_params= sess.run([self.logit,self.trans_params],feed_dict={self.Q: Q, self.A: A,self.Q_seq_vec:Q_len,self.A_seq_vec:A_len})
            verbit_seq = self.Verbit(logits=logit, batch_size=Q.shape[0], trans_params=trans_params,
                                     sequence_lengths=A_len)
            print(verbit_seq)

if __name__ == '__main__':

    init_dim=100
    batch_size=168
    Q_len=10
    A_len=50
    hidden_dim=100
    dd= Data_deal.DataDeal_1(train_path="/train1.txt", test_path="/test1.txt",
                               dev_path="/dev1.txt",
                               dim=init_dim, batch_size=batch_size, Q_len=Q_len, A_len=A_len, flag="train")
    ca=R_Net_simplify(init_dim=init_dim,hidden_dim=hidden_dim,Q_len=Q_len,A_len=A_len)
    ca.train(dd)


